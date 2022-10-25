import JuMP, GLPK

global total_counts = 0
global counts = 0

function action_info(p::CCPOMCPPlanner, b; tree_in_info=false)
    # println("Before before Rew")
    # all_rew = []
    # for i in values(reward_vectors(cm))
    #     for j in i
    #         push!(all_rew,j)
    #     end
    # end
    # println("Before Rew")
    # min_r = minimum(all_rew) #TAKE MINIMUM OF ALL REWARDS, NOT MIN MATRIX FOR SOME ACTION
    # disc = discount(p.problem) #assumes p is ConstrainedPOMDP
    # max_r = maximum(all_rew)
    # println("after")
    # ub = (max_r-min_r)/(p.solver.τ*(1-disc)) #WHAT DOES τ do?
    ub = 2.0#100.0

    λ = randn(length(p.problem.constraints))*ub
    # λ = [2.0]
    α = p.solver.α
    τ = p.solver.τ
    ν = p.solver.ν
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = CCPOMCPTree(p.problem, b, p.solver.tree_queries)

        a = search(p, b, tree, info, λ, α, τ, ν)

        # a = :A1
        p._tree = tree
        ##Add constraint update
        act_dist = p._act_dist
        sum_term = zeros(length(p.problem.constraints))
        for ha in support(act_dist)
            action = tree.a_labels[ha]
            if action != a
                sum_term += pdf(act_dist,ha).*tree.c[ha]
            end
        end
        act_to_node = Dict(tree.a_labels[tree.children[1]].=>tree.children[1])
        # @show p.solver.constraints
        # @show act_dist
        # @show (p.solver.constraints - pdf(act_dist,act_to_node[a]).*tree.c_bar[act_to_node[a]]-sum_term)
        # @show (discount(p.problem)*pdf(act_dist,act_to_node[a]))
        p.constraints = (p.constraints - pdf(act_dist,act_to_node[a]).*tree.c_bar[act_to_node[a]]-sum_term)./(discount(p.problem)*pdf(act_dist,act_to_node[a]))

        # @show p.constraints

        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    return a, info
end

action(p::CCPOMCPPlanner, b) = first(action_info(p, b))

function greedy_action(t,h,λ,ν,p)
    # t.total_greedy_actions[1] += 1
    ltn = log(t.total_n[h]+1)
    best_nodes = Vector{Int64}()
    a_ucbs = Vector{Float64}()
    node_values = Vector{Float64}()
    best_node_value = -Inf
    best_ind = 0
    node_counter = 0

    for node in t.children[h]
        node_counter += 1
        n = t.n[node]

        # if n == 0
        #     a = t.a_labels[node]
        #     w = SparseCat(node,[1.0])
        #     return [a,node,w]
        # end

        if length(λ) == 1
            node_value = t.v[node] - λ[1]*t.c[node][1] #+ p.solver.c*sqrt(ltn/(n+1))
        else
            node_value = t.v[node] - transpose(λ)*t.c[node] #+ p.solver.c*sqrt(ltn/(n+1)) #remove transpose, check sqrt,SparseCat
        end

        a_ucb = 0
        if n > 0
            node_value += p.solver.c*sqrt(ltn/(n+1))
            a_ucb += sqrt(log(n+1)/(n+1))
        end

        if node_value > best_node_value
            best_node_value = node_value
            best_ind = node_counter
        end

        push!(a_ucbs,a_ucb)
        push!(node_values,node_value)
    end

    for (ind,value) in enumerate(node_values)
        if value == node_values[best_ind]
            push!(best_nodes, t.children[h][ind])
        else
            q_diff = abs(value-node_values[best_ind])
            ucb_add = ν*(a_ucbs[ind]+a_ucbs[best_ind])
            if q_diff <= ucb_add && t.children[h][ind] ∉ best_nodes
                push!(best_nodes, t.children[h][ind])
            end
        end
    end

    if length(best_nodes) != 1
        # t.total_LP[1] += 1
        weighted_best_nodes = LP_Exact(best_nodes,λ,t.c[best_nodes],p.constraints)
        # weighted_best_nodes = LP_10(best_nodes,λ,t.c[best_nodes],p.constraints)
    elseif length(best_nodes) == 1
        weighted_best_nodes = SparseCat(best_nodes,[1.0])
    end

    ha = rand(p.rng, weighted_best_nodes) #Sample from best value nodes
    a = t.a_labels[ha]
    return [a,ha,weighted_best_nodes]
end

function search(p::CCPOMCPPlanner, b, t::CCPOMCPTree, info::Dict, λ, α, τ, ν)
    λ_count = 1
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    h = 1
    # all_rew = []
    # for i in values(reward_vectors(cm))
    #     for j in i
    #         push!(all_rew,j)
    #     end
    # end
    # min_r = minimum(all_rew) #TAKE MINIMUM OF ALL REWARDS, NOT MIN MATRIX FOR SOME ACTION
    # disc = discount(p.problem) #assumes p is ConstrainedPOMDP
    # max_r = maximum(all_rew)
    # ub = (max_r-min_r)/(τ*(1-disc)) #WHAT DOES τ do?
    # @show ub
    # ub = 20/(τ*(1-discount(p.problem)))
    ub = 2.0#100.0
    # @show p.solver.tree_queries
    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        # @show s, isterminal(p.problem, s)
        if !POMDPs.isterminal(p.problem, s)

            # rew_values = values(reward_vectors(p.problem))

            clamp!(λ,0,ub)
            if i == 1000
                p._tree = CCPOMCPTree(p.problem, b, p.solver.tree_queries)
            end
            simulate(p, s, CCPOMCPObsNode(t, 1), p.solver.max_depth, λ, ν)
            ####GREEDY POLICY HERE:
            a,ha,wb = greedy_action(t,h,λ,ν,p)
            # Constraint Implementation
            constraints = p.constraints
            # @show λ, ha, t.c[ha]


            gradient = (t.c[ha][1] - constraints[1]) < 0 ? -1 : 1
            # λ[:] = λ[:] + p.solver.α*(t.c[ha]-constraints)
            λ[:] .+= p.solver.α*gradient
            # λ = [1.0]
            λ_count += 1
            p.solver.α = 1/λ_count
            # println(λ)
            all_terminal = false
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    # h = 1
    # best_node = first(t.children[h])
    # best_v = t.v[best_node]
    # @assert !isnan(best_v)
    # for node in t.children[h][2:end]
    #     if t.v[node] >= best_v
    #         best_v = t.v[node]
    #         best_node = node
    #     end
    # end
    ####GREEDY POLICY HERE:
    a,ha,wb = greedy_action(t,h,λ,ν,p)
    p._act_dist = wb
    # println(λ)
    # println("action is $a")
    #######################
    return a#t.a_labels[best_node]
end

solve(solver::CCPOMCPSolver, c_pomdp::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = CCPOMCPPlanner(solver, c_pomdp)

function simulate(p::CCPOMCPPlanner, s, hnode::CCPOMCPObsNode, steps::Int, λ::Vector{Float64}, ν::Float64)
    if steps == 0 || isterminal(p.problem, s)
        return [0.0,zeros(length(p.problem.constraints))]
    end

    t = hnode.tree
    h = hnode.node

    # t.total_sim[1] += 1

    # ltn = log(t.total_n[h])
    # best_nodes = empty!(p._best_node_mem)
    # best_criterion_val = -Inf
    # node_values = []
    # for node in t.children[h]
    #     n = t.n[node]
    #     if n == 0 && ltn <= 0.0
    #         criterion_value = t.v[node] - transpose(λ)*t.c[node]
    #         push!(node_values,criterion_value)
    #     elseif n == 0 && t.v[node] == -Inf
    #         criterion_value = Inf
    #     else
    #         criterion_value = t.v[node] - transpose(λ)*t.c[node] + p.solver.c*sqrt(ltn/n)
    #         push!(node_values,criterion_value)
    #     end
    #     if criterion_value > best_criterion_val
    #         best_criterion_val = criterion_value
    #         empty!(best_nodes)
    #         push!(best_nodes, node)
    #     elseif criterion_value == best_criterion_val
    #         push!(best_nodes, node)
    #     end
    # end
    # # best_ind =
    # # for node in node_values
    # #     if
    # #     end
    # # end
    # weighted_best_nodes = LP_10(best_nodes)
    # ha = rand(p.rng, weighted_best_nodes) #Sample from best value nodes
    # a = t.a_labels[ha]

    ###My UCB/Action Criteria
    a,ha,wb = greedy_action(t,h,λ,ν,p)
    # a = 2
    # ha = 1
    # wb = SparseCat([1],[1.0])
    ##########################
    sp, o, r, c = ConstrainedPOMDPs.gen(p.problem, s, a, p.rng) #@gen(:sp, :o, :r)(p.problem, s, a, p.rng) #ADD COST GENERATION
    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, sp, o)
        v::Float64,c_roll::Vector{Float64} = estimate_value(p.solved_estimator,
                           p.problem,
                           sp,
                           CCPOMCPObsNode(t, hao),
                           steps-1) ######################################THIS SHOULD PROBS RETURN A COST, TOO
        # println((v,c_roll))
        R::Float64 = r + discount(p.problem)*v
        C::Vector{Float64} = c + discount(p.problem)*c_roll
    else
        R,C = [r,c] + discount(p.problem).*simulate(p, sp, CCPOMCPObsNode(t, hao), steps-1,λ,ν)
    end
    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]
    t.c[ha] += (C-t.c[ha])/t.n[ha] #Add Qcost update
    t.c_bar[ha] += (c-t.c_bar[ha])/t.n[ha] #Add c-bar update

    return [R,C]
end


function LP_10(best_nodes,λ,Qs,c_hat) #FIX ME

    K = length(c_hat)
    model = JuMP.Model(GLPK.Optimizer)
    JuMP.@variable(model, ξ1[i = 1:K] >= 0)
    JuMP.@variable(model, ξ2[i = 1:K] >= 0)
    JuMP.@variable(model, wt[i = 1:length(best_nodes)] >= 0)
    JuMP.@objective(model, Min, sum(λ[k]*(ξ1[k]+ξ2[k]) for k in 1:K))
    JuMP.@constraint(model, [sum(wt[i]*Qs[i][k] for i in 1:length(best_nodes)) for k in 1:K] .== [c_hat[k] + ξ1[k]-ξ2[k] for k in 1:K])
    JuMP.@constraint(model, sum(wt[i] for i in 1:length(best_nodes)) == 1)
    JuMP.optimize!(model)
    wts = JuMP.value.(wt)
    return SparseCat(best_nodes,wts)
end

function LP_Exact(best_nodes, λ, Qs, c_hat)
    a_min = argmin(Qs)
    a_max = argmax(Qs)
    if a_min == a_max
        a_max += 1
    end
    Q_min = Qs[a_min][1]
    Q_max = Qs[a_max][1]
    wt_a_min = 0.0
    wt_a_max = 0.0
    c_hat = c_hat[1]
    if Q_max <= c_hat
        wt_a_max = 1
    elseif Q_min >= c_hat
        wt_a_min = 1
    elseif Q_min < c_hat && c_hat < Q_max
        wt_a_min = (Q_max - c_hat)/(Q_max - Q_min)
        wt_a_max = 1.0 - wt_a_min
    end
    # @show a_min, a_max
    wts = zeros(length(best_nodes))

    wts[a_min] = wt_a_min
    wts[a_max] = wt_a_max
    # @show wts
    # @show Qs
    # @show c_hat

    return SparseCat(best_nodes, wts)
end
