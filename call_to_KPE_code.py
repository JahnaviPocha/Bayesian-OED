from juliacall import Main as jl
jl.include("inverse_prob_julia/main.jl")
rpg=jl.random_points_generator
experiments=jl.experiments
parameter_estimator=jl.parameter_estimator