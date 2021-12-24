using POMDPs, Random, POMDPModels
include("../ConstrainedPOMDPs.jl")
using .ConstrainedPOMDPs
# include("/home/bkraske/Documents/Research/JuliaCode/branch/ConstrainedPOMDPs.jl/models/ConstrainedRockSample.jl")
# using BasicPOMCP
include("CC-POMCP.jl")
using BeliefUpdaters
using CSV
using Dates
using Statistics
using DataFrames
using MarsRover

m = MarsRoverPOMDP()

function ConstrainedPOMDPs.cost(m, s, a)
    if a == 5
        return [1.0]
    elseif a > 5
        return [0.1]
    end
    return [0.0]
end

cm = Constrain(m,[4.0])
up = DiscreteUpdater(m)
b0 = initialize_belief(up,initialstate(m))


# using .CCPOMCP

# m = TigerPOMDP()
# cm = ConstrainedPOMDPs.Constrain(m,[10.0])
#
# function ConstrainedPOMDPs.cost(m,s,a)
#     if a == 0
#         return [0.0]
#     end
#     return [0.0]
# end

# planner = solve(POMCPSolver(constraints = cm.constraints;max_depth = 31,tree_queries = 10000, tree_in_info = true),cm)
# tree = POMCPTree(cm, initialstate(cm), planner.solver.tree_queries)
# onetree = simulate(planner,false,POMCPObsNode(tree,1),20,[0.0],1.0)
# info = Dict{Symbol, Any}()
# λ = zeros(length(planner.problem.constraints))
# α = 0.1
# τ = 1.0
# ν = 1.0
# act = search(planner,initialstate(cm),tree,info, λ, α, τ, ν)
# a = action(planner,initialstate(cm))
function run_the_sims()
    big_results = []
    for (x,y) in [(5,7),(7,8)]#,(15,15)]
        model = RockSamplePOMDP(x, y)
        updater = DiscreteUpdater(model)
        bel = initialize_belief(updater,initialstate(model))
    # for model in test_vec
        c_model = Constrain(model, [1.0])
        for t in [0.1,1.0]
            trial_result = main_loop(c_model,t,bel,updater)
            push!(big_results,trial_result)
        end
    end
    return big_results
end


function main_loop(m,t,b0,up)
    # av_r = 0.0
    # av_c = [0.0]
    r_mat = []
    c_mat = []
    time_mat = []
    n_mat = []
    param_mat = []
    # av_prob = zeros(length(actions(m)))
    n_runs = 100 #######################################################################
    out_loop = 0
    for _ in 1:n_runs
        out_loop += 1
        if n_runs%1 == 0
            @info out_loop, Dates.format(now(),"dd_mm_HH:MM:SS")
        end

    planner = solve(POMCPSolver(constraints = m.constraints;c= 20,max_depth = 135,tree_queries = 10000000, max_time = t, tree_in_info = true),m)
    # @info planner
    s = rand(initialstate(m))
    # up = DiscreteUpdater(m.m)
    b = deepcopy(b0)#initialize_belief(up,initialstate(m))
    planner.solver.constraints = m.constraints
    # h = 1
    r_tot = 0.0
    c_tot = zeros(length(m.constraints))
    disc = 1.0
    first = true
    steps = 0

    while !isterminal(m,s) && steps < 135
        #  @show a, planner._tree.total_n[1]
        steps += 1
        a = action(planner,b)
        # @show planner._tree.total_n[1], planner._tree.total_sim[1], planner._tree.total_greedy_actions, planner._tree.total_LP
        # break
        # @show a, planner._tree.total_n[1]
        # println("action $a")
        # println("state $s")
        act_dist = planner._act_dist
        # @show act_dist
        # act_dist = SparseCat([:A1, :A2], [0.21, 0.79])
        # act_dist = SparseCat([:A1, :A2], [0.05, 0.95])
        # a = rand(act_dist)
        # println(act_dist)
        tree::POMCPTree = planner._tree
        sp, o, r, c = c_gen(m, s, a, Random.GLOBAL_RNG)
        println(r)
        r_tot = disc.*r + r_tot
        c_tot = disc.*c + c_tot

        disc *= discount(m)
        # sum_term = zeros(length(c_hat))
        # for ha in support(act_dist)
        #     action = tree.a_labels[ha]
        #     if action != a
        #         sum_term += pdf(act_dist,action).*tree.c[ha]
        #     end
        # end
        # act_to_node = Dict(tree.a_labels[tree.children[1]].=>tree.children[1])
        # # println(tree.c_bar[act_to_node[a]])
        # # println(sum_term)
        # println(act_dist)
        # println(act_to_node[a])
        # c_hat =  planner.solver.constraints
        # planner.solver.constraints = (planner.solver.constraints - c)/discount(m)
        # c_hat = (c_hat - pdf(act_dist,act_to_node[a]).*tree.c_bar[act_to_node[a]]-sum_term)./(discount(m)*pdf(act_dist,act_to_node[a]))

        s = sp
        # println(act_to_node[a])
        # println(tree.a_labels[act_to_node[a]])
        # println(tree.o_lookup[(act_to_node[a],o)])
        # h = tree.o_lookup[(act_to_node[a],o)]
        b = update(up,b,a,o)

        # if first
        #     for a in act_dist#support(act_dist)
        #         i = actionindex(m,planner._tree.a_labels[a])
        #         # print(pdf(act_dist,a))
        #         av_prob[i] += pdf(act_dist,a)
        #     end
        #     first = false
        # end
        if steps%25 == 0
            # @show planner._tree.total_n[1]
            @info steps
        end
    end
    push!(r_mat,r_tot)
    push!(c_mat,c_tot)
    push!(time_mat,planner.solver.max_time)
    push!(n_mat,n_runs)
    # push!(param_mat,sim_param)
    # push!(depth_mat, planner.solver.max_depth)
    # push!(c_mat, planner.solver.c)

    # results = [s,r_tot,c_tot, act_dist]
    # av_r += r_tot
    # av_c += c_tot
    # av_prob += act_dist
    # return results

    end
    # av_r = av_r/n_runs
    # av_c = av_c/n_runs
    # av_prob = av_prob./n_runs
    # results2 = [av_r,av_c,n_runs]
    trial_mean_r = mean(r_mat)
    trial_stdm_r = std(r_mat)/sqrt(length(r_mat))
    trial_mean_c = mean(c_mat)
    trial_stdm_c = std(c_mat)/sqrt(length(c_mat))

    fill(trial_mean_r,n_runs)
    mc_mat = fill(trial_mean_c,n_runs)
    res_df = DataFrame([:Time=>time_mat,:N_Runs=>n_mat,:Disc_Rew=>r_mat,:Disc_Cost=>c_mat,
                        :Mean_Disc_Rew=>fill(trial_mean_r,n_runs),:Stdm_Rew=>fill(trial_stdm_r,n_runs),
                        :Mean_Disc_Cost=>fill(trial_mean_c,n_runs),:Stdm_Cost=>fill(trial_stdm_c,n_runs)])
    CSV.write("RS_Results_"*Dates.format(now(),"dd_mm_HH:MM:SS")*".csv", res_df)
    # CSV.write("RS_Params_"*Dates.format(now(),"dd_mm_HH:MM:SS")*".csv", planner)
    return res_df
end
# solver =

# using ProfileView
# ProfileView.@profview main_loop(cm,planner)
