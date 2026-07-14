mutable struct Basis{SolArrT<:AbstractArray}
    nbfaces::Int64
    nspec::Int64
    nreac::Int64
    nnodes::Int64
    blocks::Matrix{SolArrT}
end

function Basis(::Type{SolArrT}, nbfaces, nspec, nreac, nnodes) where {SolArrT<:AbstractArray}
    blocks = Matrix{SolArrT}(undef, nreac, nnodes)
    for i = 1:nreac
        for j = 1:nnodes
            blocks[i, j] = SolArrT(undef, nspec, nbfaces)
        end
    end
    Basis{SolArrT}(nbfaces, nspec, nreac, nnodes, blocks)
end

function RBS_Snapshots(main; vel, nref=2500, ratio=0.01, Nexps=3, nspec=3, St=0.0, nreac=1, inlet_MFs, P_total, T)
    d = main(nref=nref, vel=vel, RBS=true, St=St, ratio=ratio, inlet_MFs=inlet_MFs, P_total=P_total, T=T)
    grid = d[6]
    sub = d[2]


    v = [i for i = 1:num_nodes(grid)]
    sub_cat = 0 .+ view(v, sub)
    sub_out = 0 .+ view(v, d[5])

    Bas_Cat = Basis(Array{Float64}, length(sub_cat), nspec, nreac, length(sub_cat))
    Bas_Out = Basis(Array{Float64}, length(sub_out), nspec, nreac, length(sub_cat))


    for (i, v) in enumerate(sub_cat)
        snapshot = main(nref=nref, vel=vel, RBS=true, ratio=ratio, St=St, RBS_full=true, catcell=v, P_total=P_total, inlet_MFs=inlet_MFs, T=T)
        for j in 1:nspec
            Bas_Cat.blocks[1, i][j, :] = 0 .+ view(snapshot[1][j, :], snapshot[2])
            Bas_Out.blocks[1, i][j, :] = 0 .+ view(snapshot[1][j, :], snapshot[5])
        end
    end

    return Bas_Cat.blocks, Bas_Out.blocks

end


function nl_algebraic_sys_sciml!(du, u, p)
    # Unpack the parameters. Notice 'cache' and 'W' are now passed in.
    k, scale, Temp, Yin, Ncat_cells, mixture_density, molar_weights, nreacs, cache, W = p
    state_cache = get_tmp(cache, u)
    nspecies = length(Yin)

    for c in 1:Ncat_cells
        state_cache .= Yin
        W_c = W[c]
        for idx in 1:length(u)
            u_val = u[idx]
            for s in 1:nspecies
                state_cache[s] += u_val * W_c[s, idx]
            end
        end
        rates = rf(
            state_cache, k;
            T=Temp, scale=scale, density=mixture_density, molar_weights=molar_weights, RBS=true
        )
        for i in 1:nreacs
            idx = nreacs*(c-1) + i
            du[idx] = u[idx] - rates[i]
        end
    end
    return nothing
end


function compute_rbs_coeffs_n!(k; st, scale, Temp, yin, Y_snap, Ncat_cells, nreacs, Nexp=3, ei, mixture_density, molar_weights)
    Y_cat = Y_snap[1]
    Yin = yin[:, ei] .* mixture_density ./ molar_weights
    Temp
    nspecies = length(Yin)
    n_u = nreacs * Ncat_cells # Total length of `u`
    W = Vector{Matrix{Float64}}(undef, Ncat_cells)
    for c in 1:Ncat_cells
        W[c] = zeros(Float64, nspecies, n_u)
        for m in 1:Ncat_cells
            y_cat_vec = view(Y_cat[1, m], :, c)
            for j in 1:nreacs
                idx = nreacs*(m-1) + j
                # Pre-calculate and store the multiplication
                W[c][:, idx] .= y_cat_vec .* view(st, j, :)
            end
        end
    end

    T_k = eltype(k)
    chunk = ForwardDiff.pickchunksize(n_u)
    cache = DiffCache(zeros(nspecies))
    p = (k, scale, Temp[ei], Yin, Ncat_cells, mixture_density, molar_weights, nreacs, cache, W)
    max_tries=1
    rng = Xoshiro(123)
    for i in 1:max_tries
        α0 = T_k.(zeros(n_u)) #rand(rng, n_u)
        prob = NonlinearProblem(nl_algebraic_sys_sciml!, α0, p)#; lb=lower_bound,ub=upper_bound)
        sol = solve(prob, NewtonRaphson(
                autodiff=AutoForwardDiff(),
                linesearch=BackTracking()
            ); abstol=1e-10, maxiters=20)#, show_trace=Val(true))

        if SciMLBase.successful_retcode(sol)
            return reshape(sol.u, nreacs, Ncat_cells)
        end
    end
    println("solver failed !")
    return fill(0.01, nreacs, Ncat_cells)

end

function y_model(k; st, scale, nreacs, Temp, Yin, Y_snap, Ncat_cells, Exp_Index, Nexp, mixture_density, molar_weights)
    Y_cat = Y_snap[1] #Block Matrix of Snapshot for Y_cat
    Y_out = Y_snap[2]
    α = compute_rbs_coeffs_n!(k; st=st, nreacs=nreacs, scale=scale, mixture_density=mixture_density, molar_weights=molar_weights, Temp=Temp, yin=Yin, Y_snap=Y_snap, Ncat_cells=Ncat_cells, Nexp=Nexp, ei=Exp_Index)
    Y = (sum(α[i, j] .* st[i, :] .* mean(Y_out[1, j], dims=2) for i in 1:nreacs for j in 1:Ncat_cells)) .* molar_weights ./ mixture_density

    return Y
end

function nl_algebraic_sys_srom_sciml!(du, u, p)
    # Unpack the parameters from the tuple
    k, st, scale, Temp, Yin, Y_cat, Ncat_cells, mixture_density, molar_weights, nreacs, b = p

    for i in 1:nreacs
        # Calculates the residual directly into du
        du[i] = u[i] - rf(Yin .+ sum(u[j] .* b[j] .* st[j, :] for j in 1:nreacs),
            k;
            T=Temp, scale=scale, density=mixture_density, molar_weights=molar_weights, RBS=false)[i]
    end

end


function compute_srom_coeffs!(k; st, Temp, yin, b, Y_snap, Ncat_cells, scale, nreacs, Nexp=3, ei, mixture_density, molar_weights)
    Y_cat = Y_snap[1]
    Yin = yin[:, ei] .* mixture_density ./ molar_weights

    T_k = eltype(k)

    # Pack everything the residual function needs into a tuple
    p = (k, st, scale, Temp[ei], Yin, Y_cat, Ncat_cells, mixture_density, molar_weights, nreacs, b)

    # NonlinearSolve automatically applies the Implicit Function Theorem for AD here!
    sol=nothing
    max_tries=1
    rng = Xoshiro(123)
    for i in 1:max_tries
        α0 = T_k.(zeros(nreacs))
        #α0 = T_k.(rand(rng, nreacs))# Random guesses spread around 0
        prob = NonlinearProblem(nl_algebraic_sys_srom_sciml!, α0, p)#;lb=[-100.0,-100.0,1e-8], ub=[100.0, 100.0,100.0])
        sol = solve(prob, NewtonRaphson(
                autodiff=AutoForwardDiff(),
                linesearch=BackTracking()
            ); abstol=1e-10, maxiters=20)

        if SciMLBase.successful_retcode(sol)
            return sol.u
        end
    end
    println("solver failed !")
    return fill(0.1, nreacs)
end

function y_model_srom(k; st, a, b, scale, nreacs, Temp, Yin, Y_snap, Ncat_cells, Exp_Index, Nexp, mixture_density, molar_weights)
    α = compute_srom_coeffs!(k; st=st, b=b, scale=scale, nreacs=nreacs, mixture_density=mixture_density, molar_weights=molar_weights, Temp=Temp, yin=Yin, Y_snap=Y_snap, Ncat_cells=Ncat_cells, Nexp=Nexp, ei=Exp_Index)
    Y = (sum(α[i] .* st[i, :] .* a[i] for i in 1:nreacs)) .* molar_weights ./ mixture_density
    return Y
end
