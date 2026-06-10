# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:28:34 2026

@author: jahna
"""

# ============================================================
# PART 1 — IMPORTS + GLOBAL SETTINGS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from call_to_KPE_code import (
    rpg,
    experiments,
    parameter_estimator
)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

TARGET_SPECIES_INDEX = 2

TRUE_K = np.array([4000.0, 4000.0])

P_TOTAL = 50
RATIO = 0.1
STD_DATA = 1e-6
N_REPEATS = 10


# ============================================================
# COMPLETE MASS FRACTION VECTOR
# ============================================================

def complete_Y_in(Y_partial):
    last = 1.0 - np.sum(Y_partial)

    if last < 0:
        return None

    return np.append(Y_partial, last)


# ============================================================
# DECODE DESIGN VECTOR
# ============================================================

def decode_design_vector(x):
    Y_partial = x[:-1]
    Temp = x[-1]

    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, Temp


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(x):
    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        return 1e6, None

    Y_in = np.array(Y_in).reshape(NSPEC, 1)
    Temp = np.array([Temp])

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

    Y_mean = np.mean(Yexp, axis=0)[:, 0]

    y_scalar = Y_mean[TARGET_SPECIES_INDEX]

    return y_scalar, Yexp


# ============================================================
# PARAMETER ESTIMATION WRAPPER
# ============================================================

def estimate_parameters(X, Y_outputs, initial_guess=None):

    Nexps = X.shape[0]

    Y_in_all = []
    Temp_all = []

    for x in X:
        Y_in, Temp = decode_design_vector(x)
        Y_in_all.append(Y_in)
        Temp_all.append(Temp)

    Y_in_all = np.array(Y_in_all).T
    Temp_all = np.array(Temp_all)
    Y_out = np.array(Y_outputs)

    if initial_guess is None:
        initial_guess = np.array([1000.0, 1000.0])

    params = parameter_estimator(
        ratio=RATIO,
        nspec=NSPEC,
        Y_in=Y_in_all,
        Temp=Temp_all,
        P_total=P_TOTAL,
        St=np.array([[-2, -1, 2]]),
        nref=2500,
        nreac=1,
        Nexps=Nexps,
        Y_out=Y_out,
        unknown_parameters=2,
        IG=initial_guess,
        N_repeats=N_REPEATS,
        σ_data=STD_DATA,
        RBS_full=False
    )

    return np.array(params)


# ============================================================
# INITIAL DESIGN GENERATOR
# ============================================================

def generate_initial_design(N_init=3):

    X_init = np.array([
    [0.1, 0.1, 300],
    [0.5, 0.5, 600]
])

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
# CANDIDATE GENERATOR
# ============================================================

def generate_candidates(n_candidates=200):

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
# PART 2 — BOED (FIM-BASED)
# ============================================================

def compute_sensitivity(x, params, eps=1e-1):
    """
    Numerical sensitivity of outputs wrt parameters.
    Returns J matrix of shape (NSPEC, n_params)
    """

    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        return np.zeros((NSPEC, len(params)))

    Y_in = np.array(Y_in).reshape(NSPEC, 1)
    Temp = np.array([Temp])

    # Base experiment
    base = experiments(
        Y_in=Y_in,
        Temp=Temp,
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=STD_DATA,
        Nspec=NSPEC,
        k_true=np.array(params)
    )

    base_mean = np.mean(base, axis=0)[:, 0]

    J = np.zeros((NSPEC, len(params)))

    for i in range(len(params)):
        perturbed = np.array(params, dtype=float)
        perturbed[i] += eps

        y_pert = experiments(
            Y_in=Y_in,
            Temp=Temp,
            P_total=P_TOTAL,
            Nexps=1,
            ratio=RATIO,
            N_repeats=N_REPEATS,
            std_data=STD_DATA,
            Nspec=NSPEC,
            k_true=perturbed
        )

        y_pert_mean = np.mean(y_pert, axis=0)[:, 0]

        J[:, i] = ((y_pert_mean - base_mean) / eps) * params[i]

    return J


# ============================================================
# FISHER INFORMATION MATRIX
# ============================================================

def compute_fim(X, params):
    """
    Compute cumulative Fisher Information Matrix
    """
    n_params = len(params)

    F = np.zeros((n_params, n_params))

    for x in X:
        J = compute_sensitivity(x, params)
        F += J.T @ J

    return F


# ============================================================
# scale 
# ============================================================

def scale_X(X):
    """
    Scale inputs to [0,1] for stable distance comparison
    """

    X = np.array(X)
    X_scaled = np.zeros_like(X, dtype=float)

    # Scale species variables
    for i in range(NSPEC - 1):
        lb, ub = Y_BOUNDS[i]
        X_scaled[:, i] = (X[:, i] - lb) / (ub - lb)

    # Scale temperature
    lb, ub = TEMP_BOUNDS
    X_scaled[:, -1] = (X[:, -1] - lb) / (ub - lb)

    return X_scaled

# ============================================================
# SELECT NEXT EXPERIMENT (D-OPTIMAL)
# ============================================================

def select_next_experiment_oed(X_existing, params, n_candidates=20):
    """
    Choose next experiment maximizing determinant(FIM)
    """

    candidates = generate_candidates(n_candidates)

    F_current = compute_fim(X_existing, params)

    best_det = -np.inf
    best_x = None

    # scale existing once
    X_scaled = scale_X(X_existing)

    for x in candidates:

        # ==========================
        # Diversity check in scaled space
        # ==========================
        x_scaled = scale_X(np.array([x]))[0]

        min_dist = np.min(
            np.linalg.norm(X_scaled - x_scaled, axis=1)
        )

        if min_dist < 0.1:
            continue

        # ==========================
        # Compute sensitivity
        # ==========================
        J = compute_sensitivity(x, params)

        # ==========================
        # Update FIM
        # ==========================
        F_new = F_current + J.T @ J

        det_val = np.linalg.slogdet(
            F_new + 1e-8 * np.eye(len(params))
        )[1]

        if det_val > best_det:
            best_det = det_val
            best_x = x

    print(f"Best determinant gain: {best_det:.4e}")

    return best_x


# ============================================================
# STOPPING CRITERION
# ============================================================

def check_parameter_convergence(param_history, tol=1e-3):

    if len(param_history) < 2:
        return False

    delta = np.linalg.norm(
        param_history[-1] - param_history[-2]
    )

    delta_rel = delta / (np.linalg.norm(param_history[-2]) + 1e-12)

    print(f"Absolute change: {delta:.4e}")
    print(f"Relative change: {delta_rel:.4e}")

    return delta_rel < tol

# ============================================================
# MAIN BOED LOOP
# ============================================================

def bayesian_optimization(
    N_init=2,
    max_experiments=10,
    tol=1e-3
):
    """
    Sequential Bayesian Optimal Experimental Design
    """

    print("\n=== STARTING BOED ===\n")

    # ========================================================
    # INITIAL EXPERIMENTS
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

    # IMPORTANT SHAPE: (N_repeats, NSPEC, Nexps)
    Y_full = np.concatenate(Y_tensor_list, axis=2)

    # ========================================================
    # INITIAL PARAMETER ESTIMATION
    # ========================================================

    print("\nInitial parameter estimation...\n")

    params = estimate_parameters(X, Y_full)

    param_history = [params]

    print(f"Initial parameters: {params}")

    # ========================================================
    # BOED LOOP
    # ========================================================

    for iteration in range(max_experiments - N_init):

        print(f"\n--- BOED Iteration {iteration + 1} ---")

        # Select informative experiment
        x_next = select_next_experiment_oed(X, params)

        # Run experiment
        y_next, Yexp_next = run_experiment(x_next)

        print(f"New experiment:")
        print(f"x = {x_next}")
        print(f"Target species = {y_next:.6f}")

        # Append data
        X = np.vstack((X, x_next))
        y = np.append(y, y_next)
        Y_full = np.concatenate((Y_full, Yexp_next), axis=2)

        # Warm-start parameter estimation
        params = estimate_parameters(
            X,
            Y_full,
            initial_guess=params
        )

        param_history.append(params)

        print(f"Estimated parameters: {params}")

        if check_parameter_convergence(param_history, tol):
            print("\nConvergence reached → stopping early.")
            break

    print("\n=== BOED FINISHED ===\n")

    return X, y, Y_full, np.array(param_history)


# ============================================================
# PART 3 — RESULTS SUMMARY + VISUALIZATION
# ============================================================

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

    print("\nBest experiment:")
    print(f"  X = {X[best_idx]}")
    print(f"  Target species output = {y[best_idx]:.6f}")

    return final_params


# ============================================================
# PLOT 1 — EXPERIMENT DISTRIBUTION
# ============================================================

def plot_experiments(X):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        X[:, 0],
        X[:, -1],
        c=np.arange(len(X)),
        s=100
    )

    plt.colorbar(label="Experiment order")
    plt.xlabel("Y1")
    plt.ylabel("Temperature (K)")
    plt.title("BOED Experimental Design")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT 2 — PARAMETER CONVERGENCE
# ============================================================

def plot_parameter_convergence(param_history):
    params = np.array(param_history)
    exp_numbers = np.arange(1, len(params) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(exp_numbers, params[:, 0], marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("k1")
    plt.title("Convergence of k1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(exp_numbers, params[:, 1], marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("k2")
    plt.title("Convergence of k2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT 3 — TARGET SPECIES EVOLUTION
# ============================================================

def plot_target_species(y):
    plt.figure(figsize=(7, 4))

    plt.plot(y, marker='o')

    plt.xlabel("Experiment number")
    plt.ylabel("Target species output")
    plt.title("Target Species Evolution")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":

    X, y, Y_full, param_history = bayesian_optimization(
        N_init=2,
        max_experiments=12,
        tol=1e-3
    )

    final_params = summarize_results(X, y, param_history)

    plot_experiments(X)
    plot_parameter_convergence(param_history)
    plot_target_species(y)