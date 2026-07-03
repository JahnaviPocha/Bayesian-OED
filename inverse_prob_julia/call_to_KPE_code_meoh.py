# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:51:16 2026

@author: jahna
"""

from juliacall import Pkg as jlPkg

jlPkg.activate(
    "C:/Users/jahna/OneDrive/Desktop/masters/master's thesis/"
    "Bayesian-OED/inverse_prob_julia"
)
jlPkg.instantiate()

from juliacall import Main as jl

# Methanol kinetics interface. Keep this file in inverse_prob_julia next to
# the Bayesian OED Python script.
jl.include("scripts/main_meoh.jl")

rpg = jl.Example_Inverse_Problem_Validation_2D.random_points_generator
experiments = jl.Example_Inverse_Problem_Validation_2D.experiments
parameter_estimator = jl.Example_Inverse_Problem_Validation_2D.parameter_estimator