# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:36:19 2026

@author: jahna
"""

from pathlib import Path

from juliacall import Main as jl

jl.seval("import Pkg")
jlPkg = jl.Pkg

PROJECT_DIR = Path(__file__).resolve().parent

jlPkg.activate(str(PROJECT_DIR))
jlPkg.instantiate()

# Methanol kinetics interface for Fisher-information BOED.
# Keep this file in inverse_prob_julia next to the BOED Python script.
jl.include(str(PROJECT_DIR / "scripts" / "main_meoh.jl"))

module = jl.Example_Inverse_Problem_Validation_2D

rpg = module.random_points_generator
experiments = module.experiments
parameter_estimator = module.parameter_estimator

# Needed for finite-difference sensitivities dY_out/dtheta.
main_model = module.main
youts = module.youts