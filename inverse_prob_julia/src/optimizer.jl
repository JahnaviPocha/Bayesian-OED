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

function grad(f, x; method=nothing)
    if method == "autodiff"
        ∇x = ForwardDiff.gradient(f, x)
    elseif method == "finitediff"
        ∇x = FiniteDiff.finite_difference_gradient(f, x)
    else
        @error "Invalid Method for Calculating the Gradient"
    end

    return ∇x
end


function H(f, x; method=nothing)
    if method == "autodiff"
        Hx = ForwardDiff.hessian(f, x)#::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())
    #println("This prints atleast one time")
    elseif method == "finitediff"
        Hx = FiniteDiff.finite_difference_hessian(f, x)
    else
        @error "Invalid Method for Calculating the Gradient"
    end

    return Hx
end


function newton(F, x0; method="autodiff", hessian_approx=false, max_iter=100, tol=1e-13)
    x_k = x0
    x = x0
    for k in 1:max_iter
        @show res = F(x_k)
        ∇x = grad(F, x_k; method=method)

        if hessian_approx == true
            Hx = ∇x * transpose(∇x)
        else
            Hx = H(F, x_k; method=method)
        end

        #println(∇x)
        #Solve using the newton method
        x = x_k - (Hx) \ ∇x
        # x = x_k - inv(Hx) * ∇x

        # Check for convergence
        if norm(x - x_k) < tol || norm(res) < tol
            println("Converged in $k iterations")
            return x, res
        end
        x_k = x

    end

    println("Warning: Maximum iterations reached without convergence")
    return x, F(x)
end

function levenberg_marquardt(f, x0; λ=1.0e-3, x_tol=0.0, f_tol=0.0, tol=1e-18, step_size=2, max_iter=100, method="autodiff", hessian_approx=false)
    x_k = x0                     # Initialize the variables
    x = x0
    Hx = zeros(length(x0), length(x0))
    iter = 0.0
    covariance = 0.0
    for i in 1:max_iter
        fx = f(x_k)
        ∇x = grad(f, x_k; method=method)
        #J = transpose(g)
        if hessian_approx == true
            Hx = ∇x * transpose(∇x)
        else
            Hx = H(f, x_k; method=method)
        end

        LM_diag = Diagonal(∇x * transpose(∇x))

        # Levenberg-Marquardt step
        I_scaled = λ * I(size(Hx, 1))    # Scaled identity matrix
        #dx = inv(Hx + I_scaled) * (-∇x)   
        dx = (Hx + I_scaled) \ (∇x) #* fx  # Solve the damped system
        @show x = x_k - dx           # Update the solution
        @show norm(x)
        @show norm(∇x)
        @show fx


        if norm(x - x_k) < x_tol || norm(fx) < f_tol || norm(∇x) < tol
            @show norm(x - x_k)
            @show norm(∇x)
            @assert isposdef((Hx + I_scaled))
            println("Converged in $i iterations with a λ value of $λ.")
            covariance = Hx \ I(length(x0))
            @show fx
            println("and covariance matrix inverse equals $covariance.")
            iter = i
            #            println("and parameter std deviations are $(sqrt(inv(Hx)[1])) and $(sqrt(inv(Hx)[4]))")
            return x_k, iter, covariance
        end

        # Evaluate the objective function at the new point
        fx_new = f(x)
        if norm(fx_new) < norm(fx)  # If the update reduces the cost
            x_k = x                       # Accept the update
            λ /= step_size                # Reduce damping
            λ = maximum([λ, 1e-20])
        else
            λ *= step_size
            λ = minimum([λ, 1e20])              # Increase damping
        end

        @show λ

    end
    # covariance = Hx \ I(length(x0))
    println("Maximum iterations reached.")

    @show f(x_k)

    return x_k, iter, covariance

end

function nl_algebraic_sys(k, α_μ; cell_index, Temp, Yin, Y_cat, Ncat_cells)
    return α_μ[cell_index] - rf(Yin .+ sum(α_μ[j] * Y_cat[1, cell_index][:, j] for j in 1:Ncat_cells), k; T=Temp)[1]
end



function residual!(f, k; A, B, sp_index=0.0, yin, Yexp, cov=0.0, dof, T=0.0, st=0.0, print=false, Nexps=2, N_measurements=2)
    Nspecs = size(yin, 1)[1]
    NReactions = size(st, 1)
    a = mean(A)
    b = mean(B)

    yexp = Array{Float64}(undef, N_measurements, Nspecs, Nexps)
    for i in 1:Nexps
        for j in 1:Nspecs
            yexp[:, j, i] = Yexp[:, j, i] .- yin[j, i]
        end
    end

    #Calculate mean of species values over all measurements with the same experiments 
    y_mean = Matrix{Float64}(undef, Nspecs, Nexps)
    for i in 1:Nexps
        y_mean[:, i] = mean(yexp[:, :, i], dims=1)
    end

    r = y_mean ./ a
    x = copy(r)
    # if compute_uncertainity == true
    #     if lower_bound == true
    #         x = x .- sqrt(cov[1]) / a
    #         b -= std(B)
    #     else
    #         x = x .+ sqrt(cov[1]) / a
    #         b += std(B)
    #     end
    # end

    #W = inv(cov) .* a^2

    for i in 1:Nexps
        #for j in 1:Nspecs
        f[i] = sum((r[j, i] - st[1, j] * (rf(yin[:, i] .+ x[:, i] .* b, k; T=T[i])[1])) for j in 1:Nspecs)
        #end
    end

    #f(k) = sum(transpose(r[:, i] - sum(st[j, :] .* ((rf(yin[:, i] .+ x[:, i] .* b, k; T=T[i]))[j]) for j in 1:NReactions)) * W * (r[:, i] - sum(st[j, :] .* ((rf(yin[:, i] .+ r[:, i] .* b, k; T=T[i]))[j]) for j in 1:NReactions)) for i in 1:Nexps)

    #return f

end

function nl_solver(A, B, yin, Yexp; cov=0.0, dof, T=0.0, st=0.0, print=false, Nspecs=3, Nexps=2, N_measurements=2)

    k0 = ones(Nexps)
    # residual = zeros(Nspecs)
    # par=similar(k0)

    # for i in 1:Nspecs
    Root = nlsolve((F, k) -> residual!(F, k; A=A, B=B, yin=yin, Yexp=Yexp, cov=cov, dof=dof, T=T, st=st, print=print, Nexps=Nexps, N_measurements=N_measurements), k0, iterations=190, autodiff=:forward, xtol=1e-8, show_trace=false, extended_trace=true, factor=1.0, method=:trust_region) #ftol=1e-15, 
    #      residual[i] = Root.residual_norm
    par = Root.zero
    # end
    res = Root.residual_norm

    return par, res
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

    println(b)
    #Singe RBS using Forward Looping #Need to adjust ths later on 
    f_srbs_fwd(k) = sum(transpose(y_mean[:, i] - y_model_srom(k; st=st, a=a, b=b, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS, Ncat_cells=Ncat_cells, Nexp=Nexps)) * W * (y_mean[:, i] - y_model_srom(k; st=st, a=a, b=b, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS, Ncat_cells=Ncat_cells, Nexp=Nexps)) for i in 1:Nexps)[1]

    #RBS_Based function    
    f_rbs(k) = sum(transpose(y_mean[:, i] - y_model(k; st=st, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS[i], Ncat_cells=Ncat_cells, Nexp=Nexps)) * W * (y_mean[:, i] - y_model(k; st=st, nreacs=NReactions, mixture_density=mixture_density[i], molar_weights=molar_weights, Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS[i], Ncat_cells=Ncat_cells, Nexp=Nexps)) for i in 1:Nexps)[1]

    if RBS == true
        f = f_rbs
        tol = 1e-10
        solver = LBFGS(m=12)
        iter = 500
    elseif Fwd == true
        f = f_srbs_fwd
        tol = 1e-10
        iter = 500
    else
        f = f_srbs_tr
        tol = 1e-10
        solver = NelderMead()# BFGS()
        iter = 500
    end

    k0 = Initial_Guess#[4020.0, 3900.0]#ones(dof)
    # par, _ = levenberg_marquardt_nested(f, k0; T=T, y_mean=y_mean, yin=yin, Y_snap=B_RBS, Nexp=Nexps, Ncat_cells=Ncat_cells, λ=10.0, tol=1e-10, max_iter=10000, method="autodiff", hessian_approx=true)
    if lm == true #LBFGS(; m = 1, linesearch = LineSearches.HagerZhang())

        @info "Evaluating using Newton with LBFGS"
        if RBS != true
            lower = fill(0.0, 9)#[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, -1e6]#
            upper = fill(1.0, 9)#[1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, -1e-6]#fill(1e6, 8)
            inner_optimizer = NelderMead()#  outer_g_tol=tol, g_abstol=tol, g_tol=tol,
            res = optimize(f, lower, upper, k0, Fminbox(inner_optimizer), Optim.Options(iterations=iter, outer_iterations=50, store_trace=false, show_trace=true, extended_trace=false))
            #@show Optim.g_norm_trace(res)
            @show Optim.minimizer(res)
            @show Optim.minimum(res)
            @show Optim.iterations(res)
            @show Optim.x_converged(res)
            @show Optim.f_converged(res)
            @show Optim.g_converged(res)
            @show Optim.converged(res)
        else
            lower = fill(0.0, 9)#[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, -1e6]#
            upper = fill(1.0, 9)#[1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, -1e-6]#fill(1e6, 8)
            inner_optimizer = NelderMead()#  outer_g_tol=tol, g_abstol=tol, g_tol=tol,
            res = optimize(f, lower, upper, k0, Fminbox(inner_optimizer), Optim.Options(iterations=iter, outer_iterations=50, store_trace=false, show_trace=true, extended_trace=true))
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




