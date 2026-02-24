#One assumption taken is that density and temperature don't vary much 
#In calculation of mixture diff coeffs, water is also considered a non-polar molecule 
#to simplify calculation (coz all other molecules are non-polar and water has a realtively
#lower concentration)


nospec = 6
# T=273.15 #493.
# P=1.01325e6 #1e6 #1.013e6 #2.027e6 #1e6
kb = 1.380649e-16
#Values of different parameters for our Binary Diffusivity Calculation 
molar_weights = [44.01 2.016 18.01528 32.04 28.01 28.0134];
ϵkb = [244.0, 38.0, 572.4, 481.80, 98.10, 97.53]
σ = [3.763, 2.92, 2.605, 3.626, 3.65, 3.621]
μ = [0.0, 0.0, 1.844, 0.0, 0.0, 0.0]
α = [2.65, 0.79, 0.0, 0.0, 1.95, 1.76]
# Ωref=[0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10.0;
#     1.5175,1.4398,1.3204,1.2336,1.1679,1.1166,1.0753,1.0006,0.95003,0.91311,0.88453,0.84277,0.81287,0.78976,0.77111,0.75553,0.74220]



function Ωact(i, T)


    Ωref = [0.9 1.4 1.8 2.0 3.0 3.5 4.0 6.0 9.0;
        1.5175 1.2336 1.1166 1.0753 0.95003 0.91311 0.88453 0.81287 0.75553]
    A = [1 Ωref[1, i] (Ωref[1, i])^2; 1 Ωref[1, i+1] (Ωref[1, i+1])^2; 1 Ωref[1, i+2] (Ωref[1, i+2])^2]
    y = [Ωref[2, i], Ωref[2, i+1], Ωref[2, i+2]]
    a = A \ y

    return a[1] + a[2] * T + a[3] * T^2

end

function binary_diff(T, P)

    ϵjkkb = zeros(nospec, nospec) #Matrix{Float64}(undef,nospec,nospec)
    μjksq = zeros(nospec, nospec)
    σjk = zeros(nospec, nospec)
    Tjk = zeros(nospec, nospec)
    δjk = zeros(nospec, nospec)
    Ω = zeros(nospec, nospec)
    mjk = zeros(nospec, nospec)
    DB = zeros(nospec, nospec)

    #For the case of same dipole system 
    for i in 1:nospec
        for j in 1:nospec
            if j != i
                ϵjkkb[i, j] = sqrt(ϵkb[i] * ϵkb[j]) #need to vectorize all this stuff 

                μjksq[i, j] = μ[i] * μ[j]
                σjk[i, j] = 0.5 * (σ[i] + σ[j]) * 1e-8

                Tjk[i, j] = T / ϵjkkb[i, j]

                δjk[i, j] = 0.5 * μjksq[i, j]

                # println(Tjk[i,j])
                if Tjk[i, j] < 2.0
                    Ω[i, j] = Ωact(1, Tjk[i, j])
                elseif Tjk[i, j] >= 4.0
                    Ω[i, j] = Ωact(7, Tjk[i, j])
                else
                    Ω[i, j] = Ωact(4, Tjk[i, j])
                end
                mjk[i, j] = molar_weights[i] * molar_weights[j] / (6.02214e23 * (molar_weights[i] + molar_weights[j]))
                DB[i, j] = 3 * 0.0001 / 16 * sqrt(2 * pi * kb^3 * T^3 / mjk[i, j]) / (P * 1e6 * pi * σjk[i, j]^2 * Ω[i, j])
            end
        end
    end

    return DB
end



