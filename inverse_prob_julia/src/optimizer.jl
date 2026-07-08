function newton_optimizer(single_snapshot_A, single_snapshot_B, yin, Yexp; vel=0.0, Mw_avg=0.0, std_dev=1.0, scale=1.0, molar_weights=0.0, mixture_density=0.0, B_RBS=0.0, cov=0.0, dof, T=0.0, st=0.0, Fwd=false, print=false, Initial_Guess=[0.0, 0.0], lm=false, lower_bound=false, compute_uncertainity=false, RBS=true, Nexps=2, N_measurements=2)
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
    B = []
    for i in 1:Nexps
        y_mean[:, i] = mean(yexp[:, :, i], dims=1)
        A = a[i] .* st'
        push!(B, b[i] .* st')
        G[:, i] = ((A' * A) .+ 1e-12) \ A' * (y_mean[:, i] ./ molar_weights .* (mixture_density[i]))
    end

    r = y_mean


    #AA objective function
    function f_srbs_fwd(k, p)
        a, b, std_dev, Mw_avg, st, NReactions, scale, mixture_density, molar_weights, T, yin, B_RBS, Ncat_cells, Nexps, y_mean, W = p
        T_k = eltype(k)
        errors = zeros(T_k, Nexps)
        for i in 1:Nexps #  Threads.@threads :dynamic 
            y_m = y_model_srom(k;
                st=st, a=a, b=b, scale=scale, nreacs=NReactions,
                mixture_density=mixture_density[i], molar_weights=molar_weights,
                Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS, Ncat_cells=Ncat_cells, Nexp=Nexps
            )
            y_data = @view y_mean[:, i]
            diff = y_data .- y_m
            X=std_dev .* molar_weights / Mw_avg[i]
            errors[i] = sum((diff ./ X) .^ 2) #dot(diff, (W), diff) 
        end
        return sum(errors) * std_dev #* 1e-9
    end

    #PC objective function
    function f_srbs_tr(k, p)
        a, b, std_dev, Mw_avg, st, NReactions, scale, mixture_density, molar_weights, T, yin, B_RBS, Ncat_cells, Nexps, y_mean, W = p
        T_k = eltype(k)
        errors = zeros(T_k, Nexps)
        Threads.@threads :dynamic for i in 1:Nexps
            y_m=vcat(rf((yin[:, i] ./ molar_weights .* (mixture_density[i]) .+ B[i] * G[:, i]), k; T=T[i], density=mixture_density[i], molar_weights=molar_weights, RBS=true)...)
            y_data = @view G[:, i]
            diff = y_data .- y_m
            errors[i] = dot(diff,  diff) #(W_dash),
        end
        return sum(errors) #* std_dev * std_dev * 1e-6
    end

    #RBS objective function 
    function f_rbs(k, p)
        # 1. Unpack all your static data from the `p` tuple
        st, NReactions, scale, mixture_density, molar_weights, T, yin, B_RBS, Ncat_cells, Nexps, y_mean, W = p

        T_k = eltype(k)
        errors = zeros(T_k, Nexps)

        # 2. @time removed!   
        for i in 1:Nexps # Threads.@threads :dynamic
            y_m = y_model(k;
                st=st, nreacs=NReactions, scale=scale,
                mixture_density=mixture_density[i], molar_weights=molar_weights,
                Temp=T, Yin=yin, Exp_Index=i, Y_snap=B_RBS[i],
                Ncat_cells=Ncat_cells, Nexp=Nexps
            )
            y_data = @view y_mean[:, i]
            diff = y_data .- y_m
            X=std_dev .* molar_weights / Mw_avg[i]
            errors[i] = sum((diff ./ X) .^ 2)
        end
        return sum(errors) * std_dev #* 1e-7 
    end

    if RBS == true
        f = f_rbs
        tol = 1e-8
        iter=500
    elseif Fwd == true
        f = f_srbs_fwd
        tol = 1e-8
        iter=1000
    else
        f = f_srbs_tr
        tol = 1e-10
        iter=1000
    end

    k0 = collect(Initial_Guess)

    if lm == true #LBFGS(; m = 1, linesearch = LineSearches.HagerZhang())

        @info "Evaluating using Newton with LBFGS"
        if RBS != true
            if dof > 2
                lower = fill(1e-6, dof)
                upper = fill(0.999, dof)
            else
                lower = fill(1e-6, dof)
                upper = fill(20000, dof)
            end
            p = (a, b, std_dev, Mw_avg, st, NReactions, scale, mixture_density, molar_weights, T, yin, B_RBS, Ncat_cells, Nexps, y_mean, W)
            # Attach ForwardDiff
            optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
            # Create the problem (k0 is your initial guess, lb/ub are your bounds arrays)
            prob = OptimizationProblem(optf, k0, p; lb=lower, ub=upper)
            # Run Ipopt!
            sol=0.0
            println("sol at my initial guess is:", f(k0, p))
            #try
            res = solve(prob, Ipopt.Optimizer(); linear_solver="ma57",
                hessian_approximation="limited-memory", hsllib=HSL_jll.libhsl_path, print_level=5,
                tol=1e-12, max_iter=iter, nlp_scaling_method="gradient-based",
                derivative_test="first-order")
            sol=res.u
            status = res.retcode
        else
            if dof > 2
                lower = fill(1e-6, dof)
                upper = fill(0.999, dof)
            else
                lower = fill(1e-6, 2)
                upper = fill(20000, 2)
            end
            p = (st, NReactions, scale, mixture_density, molar_weights, T, yin, B_RBS, Ncat_cells, Nexps, y_mean, W)
            # Attach ForwardDiff
            optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
            prob = OptimizationProblem(optf, k0, p; lb=lower, ub=upper)
            # Run Ipopt!
            sol=0.0
            println("sol at my initial guess is:", f(k0, p))
            println("gradient at my initial guess is:", ForwardDiff.gradient(k -> f(k, p), k0))
            # k0=best_guess
            res = solve(prob, Ipopt.Optimizer(); linear_solver="ma57",
                hessian_approximation="limited-memory", hsllib=HSL_jll.libhsl_path,
                print_level=5, tol=1e-12, max_iter=iter,nlp_scaling_method="gradient-based", 
                derivative_test="first-order")
            sol=res.u
            status = res.retcode
        end
    end
    return sol, [1]
end





