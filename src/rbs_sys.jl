
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

function RBS_Snapshots(main; nref=2500, ratio=0.01, Nexps=3, nspec=3, St=0.0, nreac=1, inlet_MFs, T)
    d = main(nref=nref, RBS=true, St=St, ratio=ratio, inlet_MFs=inlet_MFs, T=T)
    grid = d[6]
    sub = d[2]


    v = [i for i = 1:num_nodes(grid)]
    sub_cat = 0 .+ view(v, sub)
    sub_out = 0 .+ view(v, d[5])

    Bas_Cat = Basis(Array{Float64}, length(sub_cat), nspec, nreac, length(sub_cat))
    Bas_Out = Basis(Array{Float64}, length(sub_out), nspec, nreac, length(sub_cat))


    for (i, v) in enumerate(sub_cat)
        snapshot = main(nref=nref, RBS=true, ratio=ratio, St=St, RBS_full=true, catcell=v, inlet_MFs=inlet_MFs, T=T)
        for j in 1:nspec
            Bas_Cat.blocks[1, i][j, :] = 0 .+ view(snapshot[1][j, :], snapshot[2])
            Bas_Out.blocks[1, i][j, :] = 0 .+ view(snapshot[1][j, :], snapshot[5])
        end
    end

    return Bas_Cat.blocks, Bas_Out.blocks

end

function nl_algebraic_sys!(F, α_μ; st, Exp_Index, Temp, Yin, Y_cat, k, Ncat_cells, mixture_density, molar_weights, nreacs=1)
    for c in 1:Ncat_cells
        for i in 1:nreacs
            # @show 3*(c-1)+i
            F[nreacs*(c-1)+i] = (α_μ[nreacs*(c-1)+i] - rf(Yin .+ sum(α_μ[nreacs*(m-1)+j] .* Y_cat[1, m][:, c] .* st[j, :] for j in 1:nreacs for m in 1:Ncat_cells), k; T=Temp, density=mixture_density, molar_weights=molar_weights, RBS=true)[i])
        end
    end
end



function compute_rbs_coeffs!(k; st, Temp, yin, Y_snap, Ncat_cells, nreacs, Nexp=3, ei, mixture_density, molar_weights) #TODO Add All reaction parameters as a struct 
    #import snapshots for every basis element if they do not change in every experiment 
    Y_cat = Y_snap[1] #Block Matrix of Snapshot for Y_cat
    α0 = zeros(nreacs * Ncat_cells)
    Yin = yin[:, ei] .* mixture_density ./ molar_weights
    Root = nlsolve((F, α_μ) -> nl_algebraic_sys!(F, α_μ; st=st, nreacs=nreacs, mixture_density=mixture_density, molar_weights=molar_weights, Exp_Index=ei, Temp=Temp[ei], Yin=Yin, Y_cat=Y_cat, k=k, Ncat_cells=Ncat_cells), α0, iterations=2000, ftol=1e-15, show_trace=false, extended_trace=false, factor=0.5, method=:trust_region) #ftol=1e-15, autodiff=:central, 
    α = reshape(Root.zero, nreacs, Ncat_cells)
    return α #returns the vector Y_out_mean mass fractions
end

function y_model(k; st, nreacs, Temp, Yin, Y_snap, Ncat_cells, Exp_Index, Nexp, mixture_density, molar_weights)
    Y_cat = Y_snap[1] #Block Matrix of Snapshot for Y_cat
    Y_out = Y_snap[2]
    α = compute_rbs_coeffs!(k; st=st, nreacs=nreacs, mixture_density=mixture_density, molar_weights=molar_weights, Temp=Temp, yin=Yin, Y_snap=Y_snap, Ncat_cells=Ncat_cells, Nexp=Nexp, ei=Exp_Index)
    Y = (sum(α[i, j] .* st[i, :] .* mean(Y_out[1, j], dims=2) for i in 1:nreacs for j in 1:Ncat_cells)) .* molar_weights ./ mixture_density

    return Y
end

function nl_algebraic_sys_srom!(F, α_μ; st, b, Temp, Yin, Y_cat, k, Ncat_cells, mixture_density, molar_weights, nreacs=1)
    for i in 1:nreacs
        F[i] = (α_μ[i] - rf(Yin .+ sum(α_μ[j] .* b[j] .* st[j, :] for j in 1:nreacs), k; T=Temp, density=mixture_density, molar_weights=molar_weights, RBS=true)[i])
    end
end

function compute_srom_coeffs!(k; st, Temp, yin, b, Y_snap, Ncat_cells, nreacs, Nexp=3, ei, mixture_density, molar_weights)
    Y_cat = Y_snap#[1]
    α0 = zeros(nreacs)
    Yin = yin[:, ei] .* mixture_density ./ molar_weights
    Root = nlsolve((F, α_μ) -> nl_algebraic_sys_srom!(F, α_μ; st=st, b=b, nreacs=nreacs, mixture_density=mixture_density, molar_weights=molar_weights, Temp=Temp[ei], Yin=Yin, Y_cat=Y_cat, k=k, Ncat_cells=Ncat_cells), α0, iterations=10000, ftol=1e-15, show_trace=false, extended_trace=false, factor=1.0, method=:trust_region) #ftol=1e-15, 
    return Root.zero
end

function y_model_srom(k; st, a, b, nreacs, Temp, Yin, Y_snap, Ncat_cells, Exp_Index, Nexp, mixture_density, molar_weights)
    #Y_out = Y_snap[2]
    α = compute_srom_coeffs!(k; st=st, b=b, nreacs=nreacs, mixture_density=mixture_density, molar_weights=molar_weights, Temp=Temp, yin=Yin, Y_snap=Y_snap, Ncat_cells=Ncat_cells, Nexp=Nexp, ei=Exp_Index)
    Y = (sum(α[i] .* st[i, :] .* a[i] for i in 1:nreacs)) .* molar_weights ./ mixture_density
    return Y
end

