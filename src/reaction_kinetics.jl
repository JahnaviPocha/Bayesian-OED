function rf(u, k; T=800.0, density=0.0, molar_weights=0.0, RBS=false)
    R = 8.31447
    # Keq1 = exp10(3066 / T - 10.592)
    # Keq2 = exp10(-2073 / T + 2.029)

    # CO2 = 1
    # H2 = 2
    # H2O = 3
    # CH3OH = 4
    # CO = 5
    # lower = [1e-2, 1e0, 1e-4, 1e-12, 1e7, 1e1, 1e1, 1e2, -2e5]
    # upper = [1e2, 1e4, 1e2, 1e-8, 1e11, 1e5, 1e5, 1e6, -2e1]
    
    # Par = lower .+ Par .* (upper .- lower)
    
   # r_meoh = Par[1] * exp(Par[6] / (R * T)) * (PP(u, CO2, T, density; RBS)) * (max(0.0,(PP(u, H2, T, density; RBS)))) * (1 - (PP(u, H2O, T, density; RBS)) * (PP(u, CH3OH, T, density; RBS)) / (Keq1 * ((PP(u, H2, T, density; RBS))^3) * (PP(u, CO2, T, density; RBS)))) / ((1 + Par[2] * (PP(u, H2O, T, density; RBS)) / (max(0.0,(PP(u, H2, T, density; RBS)))) + Par[3] * exp(Par[7] / (R * T)) * sqrt(max(0.0,(PP(u, H2, T, density; RBS)))) + Par[4]  * exp(Par[8] / (R * T)) * (PP(u, H2O, T, density; RBS)))^3)#returns RR in mol/m2-s*MtAcat 

   # r_rwgs = Par[5] * exp(Par[9] / (R * T)) * (PP(u, CO2, T, density; RBS)) * ((1 - (PP(u, H2O, T, density; RBS)) * (PP(u, CO, T, density; RBS)) / (Keq2 * (PP(u, H2, T, density; RBS)) * (PP(u, CO2, T, density; RBS))))) / (1 + Par[2] * (PP(u, H2O, T, density; RBS)) / (max(0.0,(PP(u, H2, T, density; RBS)))) + Par[3] * exp(Par[7] / (R * T)) * sqrt(max(0.0,(PP(u, H2, T, density; RBS)))) + Par[4]  * exp(Par[8] / (R * T)) * (PP(u, H2O, T, density; RBS)))#returns RR in mol/m2-s *MtAcat
    return (1.0*0.00025 * k[1] * exp(-k[2] / (R * T)) * u[1]^2 * u[2]),0.0 #Arrhenius Mass Action CO Oxidation 

    #return r_meoh, r_rwgs
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