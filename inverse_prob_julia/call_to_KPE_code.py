from juliacall import Pkg as jlPkg
from juliacall import Main as jl

# 1. Activate your local thesis environment
jlPkg.activate("C:/Users/jahna/OneDrive/Desktop/masters/master's thesis/Bayesian-OED/inverse_prob_julia")

# 2. Safely remove old references if they exist, ignoring errors if they don't
#for pkg in ["HSL", "HSL_jll"]:
 #   try:
  #      jlPkg.rm(pkg)
   # except Exception:
    #    pass  # If the package isn't found or resolved, just keep going cleanly

# 3. Develop your local source package and add the HSL interface
#jlPkg.develop(path="HSL_jll.jl.v2024.11.28")
#jlPkg.add("HSL")
jlPkg.instantiate()

# 4. Include and load your workspace models
jl.include("scripts/main.jl") 

# 5. Extract your functions
rpg = jl.Example_Inverse_Problem_Validation_2D.random_points_generator
experiments = jl.Example_Inverse_Problem_Validation_2D.experiments
parameter_estimator = jl.Example_Inverse_Problem_Validation_2D.parameter_estimator

