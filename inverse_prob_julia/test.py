# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:27:22 2026

@author: jahna
"""

import numpy as np

from call_to_KPE_code import experiments, parameter_estimator

# ============================================================
# TEST SETTINGS
# ============================================================
 #PyArray{Float64, 2, true, false, Float64}
NSPEC = 3
NEXPS = 2
N_REPEATS = 5

# ============================================================
# 1 — CREATE VALID INPUTS
# ============================================================

print("\n=== TEST 1: Generate inputs ===")

Y_in = np.array([
    [0.2, 0.25],   # species 1
    [0.3, 0.35],   # species 2
    [0.5, 0.4]     # species 3 (closure satisfied)
])

Temp = np.array([400.0, 450.0])

print("Y_in shape:", Y_in.shape)
print("Temp shape:", Temp.shape)

# ============================================================
# 2 — RUN JULIA EXPERIMENTS
# ============================================================

print("\n=== TEST 2: Running experiments ===")

Yexp = experiments(
    Y_in=Y_in,
    Temp=Temp,
    P_total=50,
    Nexps=NEXPS,
    ratio=0.1,
    N_repeats=N_REPEATS,
    std_data=1e-6,
    Nspec=NSPEC,
    k_true=[4000.0, 4000.0]
)

print("Raw Yexp shape:", np.shape(Yexp))

# ============================================================
# 3 — FORMAT FOR PARAMETER ESTIMATION
# ============================================================

print("\n=== TEST 3: Parameter estimation ===")

# Expected: (N_repeats, Nspec, Nexps)
#Yexp = np.array(Yexp)

print("Yexp shape before:", Yexp.shape)

St=[[-2, -1, 2]]
St_t = np.array(St)

IG=[0.1, 0.1]
IG_t = np.array(IG)

params = parameter_estimator(
    ratio=0.1,
    nspec=NSPEC,
    Y_in=Y_in,
    Temp=Temp,
    P_total=50,
    St =St_t,
    nref=2500,
    nreac=1,
    Nexps=NEXPS,
    Y_out=Yexp,
    unknown_parameters=2,
    IG=IG_t,
    N_repeats=N_REPEATS,
    σ_data=1e-6,
    RBS_full=False
)

print("\nEstimated parameters:", params)

print("\n=== ALL TESTS PASSED ===")