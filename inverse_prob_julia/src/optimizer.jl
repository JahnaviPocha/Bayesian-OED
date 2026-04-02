function mol_frac_to_mass_frac(mole_fractions::Vector{Float64}, molar_weights::Vector{Float64})::Vector{Float64}

    # Ensure that mole_fractions and molar_masses have the same length
    if length(mole_fractions) != length(molar_weights)
        @error ("Mole fractions and molar weights must have the same length.")
    end
    # Calculate the numerator for each component
    numerator = mole_fractions .* molar_weights
    # Calculate the denominator as the sum of all numerator values
    denominator = sum(numerator)
    # Calculate mass fractions by dividing each numerator by the denominator
    mass_fractions = numerator ./ denominator

    return mass_fractions

end


function newton_optimizer(single_snapshot_A, single_snapshot_B, yin, Yexp; molar_weights=0.0, mixture_density=0.0, B_RBS=0.0, cov=0.0, dof, T=0.0, st=0.0, Fwd=false, print=false, Initial_Guess=[0.0, 0.0], lm=false, lower_bound=false, compute_uncertainity=false, RBS=true, Nexps=2, N_measurements=2)
    Nspecs = size(yin, 1)[1]
    NReactions = size(st, 1)
    Ncat_cells = length(single_snapshot_B[1][1])
    a = [[mean(v) for v in inner] for inner in single_snapshot_A] #map(mean, single_snapshot_A)
    b = [[mean(v) for v in inner] for inner in single_snapshot_B] #map(mean, single_snapshot_B)

    yexp = Array{Float64}(undef, N_measurements, Nspecs, Nexps)
    for i in 1:Nexps
        for j in 1:Nspecs
            yexp[:, j, i] = Yexp[:, j, i] .- yin[j, i]
        end
    end

    #Calculate mean of species values over all measurements with the same experiments 
    y_mean = Matrix{Float64}(undef, Nspecs, Nexps)
    G = zeros(NReactions, Nexps)

    W = cov \ I(Nspecs)
    W_srbs = W # W .* a^2 #This definition would become obsolete
    W_dash = []
    B = []
    for i in 1:Nexps
        y_mean[:, i] = mean(yexp[:, :, i], dims=1)
        A = a[i] .* st'
        push!(B, b[i] .* st')
        G[:, i] = (A' * A) \ A' * (y_mean[:, i] ./ molar_weights .* (mixture_density[i]))
        σ_reac = (A' * A) \ A' * (sqrt.(diag(W)) ./ molar_weights .* mixture_density[i])
        push!(W_dash, (Diagonal(σ_reac) .^ 2))
    end

    r = y_mean #./ a

    f_srbs_tr(k) = sum((transpose(G[:, i] - vcat(rf((yin[:, i] ./ molar_weights .* (mixture_density[i]) .+ B[i] * G[:, i]), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true)...)) * W_dash[i] * (G[:, i] - vcat(rf((yin[:, i] ./ molar_weights .* (mixture_density[i]) .+ B[i] * G[:, i]), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true)...))) for i in 1:Nexps)
    # #Single RBS Based function  for pressure based Reactions
    f_srbs(k) = (sum(transpose((r[:, i] ./ molar_weights .* (mixture_density[i]) - sum(st[j, :] .* ((rf(((yin[:, i] .+ r[:, i] .* b) ./ molar_weights .* (mixture_density[i])), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true))[j]) for j in 1:NReactions))) * W_srbs * ((r[:, i] ./ molar_weights .* (mixture_density[i]) - sum(st[j, :] .* ((rf(((yin[:, i] .+ r[:, i] .* b) ./ molar_weights .* (mixture_density[i])), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true))[j]) for j in 1:NReactions))) for i in 1:Nexps))

    #Single RBS Based function  for Reactions not requiring Pressure Conversion  
    #f_srbs(k) = (sum(transpose((r[:, i] - sum(st[j, :] .* ((rf(((yin[:, i] .+ r[:, i] .* b)), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true))[j]) for j in 1:NReactions))) * W_srbs * ((r[:, i] - sum(st[j, :] .* ((rf(((yin[:, i] .+ r[:, i] .* b)), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true))[j]) for j in 1:NReactions))) for i in 1:Nexps))
    #Singe RBS using Forward Looping #Need to adjust ths later on 
    f_srbs_fwd(k) = sum(transpose(y_mean[:, i] - y_model_srom(k; st=st, a=a, b=b, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS, Ncat_cells=Ncat_cells, Nexp=Nexps)) * W * (y_mean[:, i] - y_model_srom(k; st=st, a=a, b=b, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS, Ncat_cells=Ncat_cells, Nexp=Nexps)) for i in 1:Nexps)[1]

    #RBS_Based function    
    function f_rbs(k)
        return sum(transpose(y_mean[:, i] - y_model(k; st=st, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS[i], Ncat_cells=Ncat_cells, Nexp=Nexps)) * W * (y_mean[:, i] - y_model(k; st=st, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS[i], Ncat_cells=Ncat_cells, Nexp=Nexps)) for i in 1:Nexps)[1]
    end
    if RBS == true
        f = f_rbs
        tol = 1e-8
        #solver = LBFGS(m=12)
        iter = 1000
    elseif Fwd == true
        f = f_srbs_fwd
        tol = 1e-8
        iter = 1000
    else
        f = f_srbs_tr
        tol = 1e-10
        #solver = NelderMead()# BFGS()
        iter = 500
    end

    k0 = Initial_Guess#[4020.0, 3900.0]#ones(dof)
    # par, _ = levenberg_marquardt_nested(f, k0; T=T, y_mean=y_mean, yin=yin, Y_snap=B_RBS, Nexp=Nexps, Ncat_cells=Ncat_cells, λ=10.0, tol=1e-10, max_iter=10000, method="autodiff", hessian_approx=true)
    if lm == true #LBFGS(; m = 1, linesearch = LineSearches.HagerZhang())

        @info "Evaluating using Newton with NelderMead"
        if RBS != true
            lower = fill(0.0, 2)#[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, -1e6]#
            upper = fill(Inf, 2)#[1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, -1e-6]#fill(1e6, 8)
            inner_optimizer = LBFGS()#  outer_g_tol=tol, g_abstol=tol, g_tol=tol,
            res = optimize(f, lower, upper, k0, Fminbox(inner_optimizer), Optim.Options(iterations=iter, outer_iterations=50, outer_g_tol=1e-8, x_abstol=1e-6, store_trace=false, show_trace=true, extended_trace=false))
            #@show Optim.g_norm_trace(res)
            @show Optim.minimizer(res)
            @show Optim.minimum(res)
            @show Optim.iterations(res)
            @show Optim.x_converged(res)
            @show Optim.f_converged(res)
            @show Optim.g_converged(res)
            @show Optim.converged(res)
        else
            lower = fill(0.0, 2)#[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, -1e6]#
            upper = fill(Inf, 2)#[1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, -1e-6]#fill(1e6, 8)
            inner_optimizer = LBFGS()#  outer_g_tol=tol, g_abstol=tol, g_tol=tol,
            res = optimize(f, lower, upper, k0, Fminbox(inner_optimizer), Optim.Options(iterations=iter, outer_iterations=50, x_abstol=1e-5, g_abstol=1e-5, store_trace=false, show_trace=true, extended_trace=false))
            # @show Optim.g_norm_trace(res)
            @show Optim.iterations(res)
            @show Optim.x_converged(res)
            @show Optim.f_converged(res)
            @show Optim.g_converged(res)
            @show Optim.converged(res)
            # @show Optim.f_trace(res)
            @show Optim.minimizer(res)
            @show Optim.minimum(res)
            #@show @time par, iter, covariance = levenberg_marquardt(f, k0; λ=1.0, step_size=10.0, tol=tol, max_iter=1000, method="finitediff", hessian_approx=false)
        end
    else
        covariance = 0.0
        par = trust_region(k0, f, grad, H; tol=tol, method="cauchy")
        #par, _ = newton(f, k0; max_iter=10000, tol=tol, method="finitediff")
    end
    return res.minimizer, [1]#par, covariance #
end




