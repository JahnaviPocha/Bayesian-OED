# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:51:16 2026

@author: jahna
"""

from pathlib import Path

from juliacall import Main as jl

jl.seval("import Pkg")
jlPkg = jl.Pkg

PROJECT_DIR = Path(__file__).resolve().parent

jlPkg.activate(str(PROJECT_DIR))
jlPkg.instantiate()

# Methanol kinetics interface. Keep this file in inverse_prob_julia next to
# the Bayesian OED Python script.
jl.include(str(PROJECT_DIR / "scripts" / "main_meoh.jl"))

rpg = jl.Example_Inverse_Problem_Validation_2D.random_points_generator
experiments = jl.Example_Inverse_Problem_Validation_2D.experiments
parameter_estimator = jl.Example_Inverse_Problem_Validation_2D.parameter_estimator

# One self-contained random-design study at a given experiment count and noise
# level: it builds the Halton design, generates the data and estimates the
# parameters, with only Nexps and std_data varying. random_meoh_sweep.py runs
# each (Nexps, noise) pair as an independent task on that basis.
complete_workflow = jl.Example_Inverse_Problem_Validation_2D.complete_workflow