from juliacall import Pkg as jlPkg
jlPkg.activate("C:/Users/jahna/OneDrive/Desktop/masters/master's thesis/Bayesian-OED/inverse_prob_julia")
jlPkg.instantiate()
#jlPkg.add("PythonCall") 
#jlPkg.add("LaTeXStrings")
from juliacall import Main as jl
jl.include("scripts/main.jl")
rpg=jl.Example_Inverse_Problem_Validation_2D.random_points_generator
experiments=jl.Example_Inverse_Problem_Validation_2D.experiments
parameter_estimator=jl.Example_Inverse_Problem_Validation_2D.parameter_estimator


