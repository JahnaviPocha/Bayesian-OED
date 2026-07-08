function rf(u, Par; scale=1.0, T=800.0, density=0.0, molar_weights=0.0, RBS=false)
    R = 8.31447
    if length(Par) > 2
        Keq1 = exp10(3066 / T - 10.592)
        Keq2 = exp10(-2073 / T + 2.029)

        CO2 = 1
        H2 = 2
        H2O = 3
        CH3OH = 4
        CO = 5

        lower=[0.1,0.1,0.1,0.1,0.1,1e4,1e4,1e4,1e4] #fill(0.1, 9)#[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1.5e-1]
        upper=[1e5, 1e4, 1e5, 1e5, 1e5, 1.5e5, 1.5e5, 1.5e5, 1.5e5]
        Par = exp10.(log10.(lower) .+ Par .* (log10.(upper) .- log10.(lower)))

        Tref=501.57
        # 1. Evaluate partial pressures exactly ONCE and cache them
        p_CO2 = smooth_positive(PP(u, CO2, T, density; RBS=RBS))
        p_H2O = smooth_positive(PP(u, H2O, T, density; RBS=RBS))
        p_CH3OH = smooth_positive(PP(u, CH3OH, T, density; RBS=RBS))
        p_CO = smooth_positive(PP(u, CO, T, density; RBS=RBS))
        p_H2 = smooth_positive(PP(u, H2, T, density; RBS=RBS))


        inv_T_diff_over_R = (1.0 / Tref - 1.0 / T) / R

        # Pre-calculate the exponential factors
        exp_k1 = exp((-Par[6] * inv_T_diff_over_R))
        exp_k2 = exp((Par[9] * inv_T_diff_over_R))
        exp_ad1 = exp((-Par[7] * inv_T_diff_over_R))
        exp_ad2 = exp((-Par[8] * inv_T_diff_over_R))

        denom_base = 1.0 +
                     Par[2] * (p_H2O / p_H2) +
                     Par[3] * exp_ad1 * sqrt(p_H2) +
                     Par[4] * exp_ad2 * p_H2O
        safe_denom_base = smooth_positive(denom_base)
        # 4. Calculate final rates using the cached variables
        r_meoh = Par[1] * exp_k1 * p_CO2 * p_H2 * (1.0 - (p_H2O * p_CH3OH) / (Keq1 * (p_H2^3) * p_CO2 + 1e-8)) /
                 (safe_denom_base^3)

        r_rwgs = Par[5] * exp_k2 * p_CO2 * (1.0 - (p_H2O * p_CO) / (Keq2 * p_H2 * p_CO2 + 1e-8)) /
                 safe_denom_base

        return r_meoh, r_rwgs

    else
        return (scale * 0.00025 * Par[1] * exp(-Par[2] / (R * T)) * u[1]^2 * u[2]), 0.0 #Arrhenius Mass Action CO Oxidation 
    end
end

function smooth_positive(x; lower_bound=1e-2, eps=1e-4)
    y = x - lower_bound
    sq = sqrt(y^2 + eps)
    # Stable computation of: y + sqrt(y^2 + eps)
    if y < 0
        # When y is negative, we use the conjugate form to avoid catastrophic cancellation.
        # This guarantees the eps is never absorbed by floating-point limits.
        val = eps / (sq - y)
    else
        val = y + sq
    end

    return 0.5 * val + lower_bound
end

Rbar = 8.31447e-5
molar_weights = [44.01, 2.016, 18.01528, 32.04, 28.01, 28.0134]

function PP(Y, i, T, density; RBS)
    if RBS == true
        pp = Y[i] * Rbar * T
    else
        pp = Y[i] * density * Rbar * T / molar_weights[i]
    end

    #@show (typeof(pp))
    return pp#10*Y[i]*M/molar_weights[i] #201.16 *3.235 Need to check the avg molar weight 
end