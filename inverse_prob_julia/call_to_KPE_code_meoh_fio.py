# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:36:19 2026

@author: jahna
"""

from juliacall import Pkg as jlPkg

jlPkg.activate(
    "C:/Users/jahna/OneDrive/Desktop/masters/master's thesis/"
    "Bayesian-OED/inverse_prob_julia"
)
jlPkg.instantiate()

from juliacall import Main as jl

# Methanol kinetics interface for Fisher-information BOED.
# Keep this file in inverse_prob_julia next to the BOED Python script.
jl.include("scripts/main_meoh.jl")

module = jl.Example_Inverse_Problem_Validation_2D

rpg = module.random_points_generator
experiments = module.experiments
parameter_estimator = module.parameter_estimator

# Needed for finite-difference sensitivities dY_out/dtheta.
main_model = module.main
youts = module.youts