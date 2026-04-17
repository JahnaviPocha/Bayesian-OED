# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:26:47 2026

@author: jahna
"""

# ============================================================
# PART 1 — DESIGN SPACE + JULIA INTERFACE (GENERALIZED)
# ============================================================

import numpy as np

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
# For methanol (3rd species)
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

def run_experiment(x):
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
        std_data=STD_DATA,
        Nspec=NSPEC,
        k_true=TRUE_K
    )

    # Average over repeats
    Y_mean = np.mean(Yexp, axis=0)[:, 0]

    # Select target species (methanol)
    y_scalar = Y_mean[TARGET_SPECIES_INDEX]

    return y_scalar, Yexp


# ============================================================
# PARAMETER ESTIMATION WRAPPER
# ============================================================

from juliacall import Main as jl

def estimate_parameters(X, Y_outputs):
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

    Y_out_reduced = np.transpose(Y_outputs, (1, 2, 0))  
    # (NSPEC, Nexps, N_repeats)

    # ================================
    # CONVERT TO JULIA ARRAYS
    # ================================
    Y_in_jl = jl.Array(Y_in_all)
    Temp_jl = jl.Array(Temp_all)
    Y_out_jl = jl.Array(Y_out_reduced)

    St_jl = jl.Array([[-2, -1, 2]])
    IG_jl = jl.Array([0.1, 0.1])

    # ================================
    # DEBUG (KEEP THIS)
    # ================================
    print("\nDEBUG (FINAL FIX):")
    print("Y_in:", Y_in_all.shape)          # (1, Nexps)
    print("Temp:", Temp_all.shape)          # (Nexps,)
    print("Y_out:", Y_out_reduced.shape)    # (1, Nexps, N_repeats)

    # ================================
    # CALL JULIA
    # ================================
    params = parameter_estimator(
        ratio=RATIO,
        nspec=NSPEC,        # ✔ keep 3 (chemistry)
        Y_in=Y_in_jl,
        Temp=Temp_jl,
        P_total=P_TOTAL,
        St=St_jl,
        nref=2500,
        nreac=1,
        Nexps=Nexps,
        Y_out=Y_out_jl,
        unknown_parameters=2,
        IG=IG_jl,
        N_repeats=N_REPEATS,
        σ_data=STD_DATA,
        RBS_full=False
    )

    return np.array(params)

# ============================================================
# INITIAL DESIGN GENERATOR
# ============================================================

def generate_initial_design(N_init=3):
    """
    Generate valid initial BO points
    """

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
# PART 2 — GAUSSIAN PROCESS + EXPECTED IMPROVEMENT
# ============================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm


# ============================================================
# HELPER: SCALE INPUT SPACE (IMPORTANT)
# ============================================================

def scale_X(X):
    """
    Scale inputs to [0, 1] for GP stability
    """

    X_scaled = np.zeros_like(X)

    # Scale species
    for i in range(NSPEC - 1):
        lb, ub = Y_BOUNDS[i]
        X_scaled[:, i] = (X[:, i] - lb) / (ub - lb)

    # Scale temperature
    lb, ub = TEMP_BOUNDS
    X_scaled[:, -1] = (X[:, -1] - lb) / (ub - lb)

    return X_scaled


# ============================================================
# BUILD GP MODEL
# ============================================================

def build_gp_model(X, y):
    """
    Train Gaussian Process surrogate
    """

    X_scaled = scale_X(X)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,              # noise level (important)
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp.fit(X_scaled, y)

    return gp


# ============================================================
# EXPECTED IMPROVEMENT (EI)
# ============================================================

def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    """
    Compute Expected Improvement

    X_candidates : (N, dim)
    gp           : trained GP
    y_best       : best observed value
    """

    X_scaled = scale_X(X_candidates)

    mu, sigma = gp.predict(X_scaled, return_std=True)
    sigma = sigma.reshape(-1, 1)

    mu = mu.reshape(-1, 1)

    with np.errstate(divide='warn'):
        improvement = mu - y_best - xi   # maximize methanol
        Z = improvement / sigma

        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.ravel()


# ============================================================
# GENERATE CANDIDATE POINTS
# ============================================================

def generate_candidates(n_candidates=500):
    """
    Random candidate sampling in design space
    """

    candidates = []

    while len(candidates) < n_candidates:
        Y_partial = [
            np.random.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]

        Temp = np.random.uniform(*TEMP_BOUNDS)

        x = np.array(Y_partial + [Temp])

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            candidates.append(x)

    return np.array(candidates)


# ============================================================
# SELECT NEXT EXPERIMENT
# ============================================================

def select_next_experiment(gp, X, y, n_candidates=500):
    """
    Choose next experiment using EI
    """

    X_candidates = generate_candidates(n_candidates)

    y_best = np.max(y)   # maximizing methanol

    ei = expected_improvement(X_candidates, gp, y_best)

    best_idx = np.argmax(ei)

    return X_candidates[best_idx]


# ============================================================
# PART 3 — BAYESIAN OPTIMIZATION LOOP + STOPPING
# ============================================================

def check_parameter_convergence(param_history, tol=1e-3):
    """
    Stop if parameters stop changing
    """

    if len(param_history) < 2:
        return False

    delta = np.linalg.norm(
        param_history[-1] - param_history[-2]
    )

    print(f"Parameter change: {delta:.4e}")

    return delta < tol


# ============================================================
# MAIN BO LOOP
# ============================================================

def bayesian_optimization(
    N_init=3,
    max_experiments=10,
    tol=1e-3
):
    """
    Full BOED loop

    Objective:
    - Minimize experiments
    - Maximize information (parameter convergence)
    """

    print("\n=== STARTING BAYESIAN OPTIMIZATION ===\n")

    # ========================================================
    # STEP 1 — INITIAL EXPERIMENTS
    # ========================================================

    X = generate_initial_design(N_init)

    y_list = []
    Y_tensor_list = []

    print("Running initial experiments...\n")

    for i, x in enumerate(X):
        y_scalar, Yexp = run_experiment(x)

        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)

        print(f"Init Exp {i+1}: y = {y_scalar:.6f}")

    y = np.array(y_list)
    Y_full = np.concatenate(Y_tensor_list, axis=2)   # shape (Nspec, N)

    # ========================================================
    # STEP 2 — INITIAL PARAMETER ESTIMATION
    # ========================================================

    print("\nInitial parameter estimation...\n")

    params = estimate_parameters(X, Y_full)
    param_history = [params]

    print(f"Initial parameters: {params}")

    # ========================================================
    # STEP 3 — BO LOOP
    # ========================================================

    for iteration in range(max_experiments - N_init):

        print(f"\n--- BO Iteration {iteration + 1} ---")

        # 3.1 Train GP
        gp = build_gp_model(X, y)

        # 3.2 Select next experiment
        x_next = select_next_experiment(gp, X, y)

        # 3.3 Run experiment (Julia CFD)
        y_next, Yexp_next = run_experiment(x_next)

        print(f"New experiment:")
        print(f"  x = {x_next}")
        print(f"  Methanol = {y_next:.6f}")

        # 3.4 Append data
        X = np.vstack((X, x_next))
        y = np.append(y, y_next)
        Y_full = np.concatenate((Y_full, Yexp_next), axis=2)

        # 3.5 Parameter estimation
        params = estimate_parameters(X, Y_full)
        param_history.append(params)

        print(f"Estimated parameters: {params}")

        # 3.6 Stopping criterion (IMPORTANT)
        if check_parameter_convergence(param_history, tol):
            print("\nConvergence reached → stopping early.")
            break

    print("\n=== BO FINISHED ===\n")

    return X, y, Y_full, np.array(param_history)


# ============================================================
# PART 4 — RESULTS SUMMARY + VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_results(X, y, param_history):
    print("\n===== FINAL RESULTS SUMMARY =====\n")

    print(f"Total experiments used: {len(X)}")

    final_params = param_history[-1]

    print("\nFinal estimated parameters:")
    print(f"  k1 = {final_params[0]:.6f}")
    print(f"  k2 = {final_params[1]:.6f}")

    best_idx = np.argmax(y)

    print("\nBest experiment (max methanol):")
    print(f"  X = {X[best_idx]}")
    print(f"  Methanol = {y[best_idx]:.6f}")

    return final_params


# ============================================================
# PLOT 1 — EXPERIMENT DISTRIBUTION
# ============================================================

def plot_experiments(X):
    plt.figure()

    # plot first species vs temperature
    plt.scatter(X[:, 0], X[:, -1], c=np.arange(len(X)), s=80)

    plt.colorbar(label="Experiment order")
    plt.xlabel("Y1 (species 1)")
    plt.ylabel("Temperature (K)")
    plt.title("Experimental Design (BOED)")
    plt.grid(True)

    plt.show()


# ============================================================
# PLOT 2 — PARAMETER CONVERGENCE
# ============================================================

def plot_parameter_convergence(param_history):
    params = np.array(param_history)
    exp_numbers = np.arange(1, len(params) + 1)

    # k1
    plt.figure()
    plt.plot(exp_numbers, params[:, 0], marker='o')
    plt.xlabel("Experiment number")
    plt.ylabel("k1")
    plt.title("Parameter Convergence: k1")
    plt.grid(True)
    plt.show()

    # k2
    plt.figure()
    plt.plot(exp_numbers, params[:, 1], marker='s')
    plt.xlabel("Experiment number")
    plt.ylabel("k2")
    plt.title("Parameter Convergence: k2")
    plt.grid(True)
    plt.show()


# ============================================================
# PLOT 3 — METHANOL EVOLUTION
# ============================================================

def plot_methanol(y):
    plt.figure()

    plt.plot(y, marker='o')

    plt.xlabel("Experiment number")
    plt.ylabel("Methanol fraction")
    plt.title("Methanol Production Across Experiments")

    plt.grid(True)
    plt.show()


# ============================================================
# MAIN DRIVER (RUN EVERYTHING)
# ============================================================

if __name__ == "__main__":

    # Run BOED
    X, y, Y_full, param_history = bayesian_optimization(
        N_init=3,
        max_experiments=8,
        tol=1e-3
    )

    # Summary
    final_params = summarize_results(X, y, param_history)

    # Plots
    plot_experiments(X)
    plot_parameter_convergence(param_history)
    plot_methanol(y)