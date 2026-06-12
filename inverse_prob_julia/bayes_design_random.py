# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:48:52 2026

@author: jahna
"""


# ============================================================
# PART 1 — DESIGN SPACE + JULIA INTERFACE (GENERALIZED)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IMPORT JULIA INTERFACE (your KPE bridge)
# ============================================================

from call_to_KPE_code import (
    rpg,
    experiments,
    parameter_estimator
)

# ============================================================
# GLOBAL SETTINGS (EASY TO MODIFY)
# ============================================================

NSPEC = 3   # number of species (change anytime)

# Bounds from Julia main.jl
# lb = [0.1, 0.1, 300]
# ub = [0.5, 0.5, 600]

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

# Target species index (0-based Python indexing)
# For output (3rd species)
TARGET_SPECIES_INDEX = 2

# True kinetic parameters (used in experiments)
TRUE_K = [4000.0, 4000.0]

# Fixed parameters (same as Julia)
P_TOTAL = 50
RATIO = 0.1
STD_DATA = 1e-6
N_REPEATS = 10


# ============================================================
# HELPER: COMPLETE MASS FRACTION VECTOR
# ============================================================

def complete_Y_in(Y_partial):
    """
    Convert partial species vector → full Y_in
    Enforces sum(Y) = 1
    """
    last = 1.0 - np.sum(Y_partial)
    
    # Safety check
    if last < 0:
        return None
    
    return np.append(Y_partial, last)


# ============================================================
# HELPER: CONVERT BO VECTOR → PHYSICAL INPUT
# ============================================================

def decode_design_vector(x):
    """
    x = [Y1, Y2, ..., Y(n-1), Temp]
    """
    Y_partial = x[:-1]
    Temp = x[-1]

    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, Temp


# ============================================================
# EXPERIMENT WRAPPER (CRITICAL)
# ============================================================

def run_experiment(x, noise_level):
    """
    Runs ONE experiment via Julia CFD model

    Returns:
    --------
    y_scalar : float
        target species output (for GP)
    y_full   : full species output (for analysis)
    """

    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        return 1e6, None  # penalize invalid region

    # Reshape for Julia call
    Y_in = np.array(Y_in).reshape(NSPEC, 1)
    Temp = np.array([Temp])

    # Run experiment
    Yexp = experiments(
        Y_in=Y_in,
        Temp=Temp,
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
        k_true=TRUE_K
    )
    
    print("Yexp type:", type(Yexp))
    print("Yexp shape:", np.array(Yexp).shape)

    # Average over repeats
    Y_mean = np.mean(Yexp, axis=0)[:, 0]

    # Select target species (output)
    y_scalar = Y_mean[TARGET_SPECIES_INDEX]

    return y_scalar, Yexp


# ============================================================
# PARAMETER ESTIMATION WRAPPER
# ============================================================

#from juliacall import Main as jl

def estimate_parameters(X, Y_outputs, noise_level):
    """
    Calls Julia parameter estimator (FINAL CORRECT VERSION)
    """

    Nexps = X.shape[0]

    # ================================
    # BUILD INPUTS
    # ================================
    Y_in_all = []
    Temp_all = []

    for x in X:
        Y_in, Temp = decode_design_vector(x)

        # ✔ keep only independent variable (solver reality)
        Y_in_all.append(Y_in)   # ✅ FIXED
        Temp_all.append(Temp)

    Y_in_all = np.array(Y_in_all).T        # (NSPEC, Nexps)
    Temp_all = np.array(Temp_all)          # (Nexps,)

    Y_out = np.array(Y_outputs)

    # ================================
    # DEBUG (KEEP THIS)
    # ================================
    print("\nDEBUG (FINAL FIX):")
    print("Y_in:", Y_in_all.shape)          # (1, Nexps)
    print("Temp:", Temp_all.shape)          # (Nexps,)
    print("Y_out:", Y_out.shape)    # (1, Nexps, N_repeats)

    # ================================
    # CALL JULIA
    # ================================
    params = parameter_estimator(
        ratio=RATIO,
        nspec=NSPEC,        # ✔ keep 3 (chemistry)
        Y_in=Y_in_all,
        Temp=Temp_all,
        P_total=P_TOTAL,
        St=np.array([[-2, -1, 2]]),
        nref=2500,
        nreac=1,
        Nexps=Nexps,
        Y_out=Y_out,
        unknown_parameters=2,
        IG=np.array([1000.0, 1000.0]),
        N_repeats=N_REPEATS,
        σ_data=noise_level,
        RBS_full=True
    )
    
    
    print("params type:", type(params))
    print("params value:", np.array(params))
    print("params shape:", np.array(params).shape)

    return np.array(params)

# ============================================================
# RANDOM DESIGN LOOP
# ============================================================


def generate_initial_design(N_init=8):

    X_init = []

    while len(X_init) < N_init:

        Y_partial = [
            np.random.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]

        Temp = np.random.uniform(*TEMP_BOUNDS)

        x = np.array(Y_partial + [Temp])

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            X_init.append(x)

    return np.array(X_init)

# ============================================================
# RANDOM DESIGN LOOP
# ============================================================

def random_design_study(
    N_experiments=8
):
    """
    Pure random experimental design.

    Same design space as BOED.
    Same parameter estimator.
    """

    print("\n=== RANDOM DESIGN STUDY ===\n")

    # --------------------------------
    # Generate random experiments
    # --------------------------------

    X = generate_initial_design(N_experiments)

    y_list = []
    Y_tensor_list = []

    for i, x in enumerate(X):

        print(f"\nExperiment {i+1}/{N_experiments}")

        y_scalar, Yexp = run_experiment(
            x,
            noise_level=STD_DATA
            )

        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)

        print(f"Output = {y_scalar:.6f}")

    y = np.array(y_list)

    Y_full = np.concatenate(
        Y_tensor_list,
        axis=2
    )

    # --------------------------------
    # Parameter estimation
    # --------------------------------

    print("\nEstimating parameters...\n")

    params = estimate_parameters(
        X,
        Y_full,
        noise_level=STD_DATA
    )

    print("\nFinal estimated parameters:")
    print(params)

    return X, y, Y_full, params

# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================

def summarize_results(X, y, params):

    print("\n===== FINAL RESULTS SUMMARY =====\n")

    print(f"Total experiments used: {len(X)}")

    print("\nFinal estimated parameters:")
    print(f"  k1 = {params[0]:.6f}")
    print(f"  k2 = {params[1]:.6f}")

    best_idx = np.argmax(y)

    print("\nBest experiment (max output):")
    print(f"  X = {X[best_idx]}")
    print(f"  Output = {y[best_idx]:.6f}")

    return params

# ============================================================
# PARAMETER ERROR FUNCTION
# ============================================================



def parameter_error(params):

    true_params = np.array([4000.0, 4000.0])

    error_percent = (
        np.linalg.norm(params - true_params)
        /
        np.linalg.norm(true_params)
    ) * 100

    print(
        f"\nRelative parameter error = "
        f"{error_percent:.2f}%"
    )

    return error_percent


# ============================================================
# RANDOM DESIGN PLOT
# ============================================================


def plot_random_design(X):

    plt.figure()

    plt.scatter(
        X[:,0],
        X[:,-1],
        c=np.arange(len(X)),
        s=100
    )

    plt.colorbar(
        label="Experiment Number"
    )

    plt.xlabel("Y1")
    plt.ylabel("Temperature")

    plt.title(
        "Random Experimental Design"
    )

    plt.grid(True)
    plt.show()

# ============================================================
# OUTPUT EVOLUTION PLOT
# ============================================================

def plot_output(y):

    plt.figure()

    plt.plot(y, marker='o')

    plt.xlabel("Experiment Number")
    plt.ylabel("Output")

    plt.title("Output Across Random Experiments")

    plt.grid(True)
    plt.show()
    
    
# ============================================================
# OUTPUT EVOLUTION PLOT
# ============================================================

if __name__ == "__main__":

    X, y, Y_full, params = random_design_study(
        N_experiments=8
    )

    summarize_results(X, y, params)

    parameter_error(params)

    plot_random_design(X)

    plot_output(y)    
    
    