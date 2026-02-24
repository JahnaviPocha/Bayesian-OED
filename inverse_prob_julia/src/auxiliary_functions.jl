Rbar = 8.31447e-5

function Y_to_P(Y, density, T, molar_weights) #Converting the Mass fractions to partial pressures using Ideal Gas Law

    Partial_Pressures=zeros(6)
    for i in 1:6
        Partial_Pressures[i] = Y[i] * density * Rbar * T / molar_weights[i]
    end

    return Partial_Pressures # Returns a vector of partial pressures of species 
end

function reaction_parameters(A, B; T) #Computing reaction parameters based on Arrhenius equation 
    return A .* exp.(B / (Rbar * 1e5 * T))
end

function average_molar_weight(mass_fracs, molar_weights)
    preM = (mass_fracs ./ molar_weights) #Average molecular weight of the mixture 
    Mwavg = 1 / sum(preM)
    return Mwavg
end

function density_rechner(molar_weights, mass_fracs, P_total, T_inlet)
    Mwavg = average_molar_weight(mass_fracs, molar_weights)
    mixture_density = (P_total * Mwavg) / (Rbar * T_inlet)
    return mixture_density
end

function youts(d; Nspec=6)
    Yout = zeros(Nspec)
    for i in 1:Nspec
        yout = 0 .+ view(d[1][i, :], d[5])
        Yout[i] = mean(yout)
    end
    return Yout
end

function youts_rom(d, r)
    Yout = zeros(3)
    for i in 1:3
        yout = 0 .+ view(r[i, :], d[5])
        Yout[i] = mean(yout)
    end
    return Yout
end

function ycat(d; Nspec=6)
    Ycat = zeros(Nspec)
    for i in 1:Nspec
        ycat = 0 .+ view(d[1][i, :], d[2])
        Ycat[i] = mean(ycat)
    end
    return Ycat
end

function face_vels(bfvel, bfaceindex)
    e = zeros(length(bfaceindex) + 1)
    #println(size(bfvel))
    #println(size(bfaceindex))
    e[1] = bfvel[1, bfaceindex[1]]
    @show bfaceindex[1]
    for (i, v) in enumerate(bfaceindex)
        if i < length(bfaceindex)
            e[i+1] = bfvel[2, bfaceindex[i]] + bfvel[1, bfaceindex[i+1]]
        end
    end
    e[end] = bfvel[2, length(bfaceindex)+1]

    return e
end

function cell_concentrations(sol, sp_index)
    yout = 0 .+ view(sol[1][sp_index, :], sol[5])
    return yout
end

function yout_weighted(sol; Nspecs=3)
    bface_index = unique(sol[8])
    face_vel = face_vels(sol[7], bface_index)
    weighted_integral = zeros(Nspecs)
    for i in 1:Nspecs
        yout = cell_concentrations(sol, i)
        integral_terms = yout .* face_vel
        num = yout .* integral_terms
        println(sum(integral_terms) * sol[10])
        weighted_integral[i] = sum(num) / sum(integral_terms)
    end

    return abs.(weighted_integral)
end



function yout_mean(d; nspecs=1)
    Yout = zeros(nspecs)
    for i in 1:nspecs
        yout = 0 .+ view(d[1][i, :], d[5])
        Yout[i] = mean(yout)
    end
    return Yout
end

function ycats(d)
    Ycat = zeros(3)
    #for i in 1:3
    Ycat = 0 .+ view(d[1][1, :], d[2])

    #end
    return Ycat
end