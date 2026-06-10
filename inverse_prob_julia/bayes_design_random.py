# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:48:52 2026

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
STD_DATA = 1e-4
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
        RBS_full=True
    )

    return np.array(params)


# ============================================================
# INITIAL DESIGN GENERATOR
# ============================================================

def generate_initial_design(N_init=2):

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

def generate_candidates(n_candidates=50):

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
# RANDOM DESIGN BASELINE
# ============================================================

def random_design(
    N_init=2,
    max_experiments=12
):

    print("\n=== RANDOM DESIGN ===\n")

    X = generate_initial_design(N_init)

    y_list = []
    Y_tensor_list = []

    for x in X:
        y_scalar, Yexp = run_experiment(x)

        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)

    y = np.array(y_list)
    Y_full = np.concatenate(Y_tensor_list, axis=2)

    params = estimate_parameters(X, Y_full)

    param_history = [params]

    for iteration in range(max_experiments - N_init):

        print(f"\n--- RANDOM Iteration {iteration+1} ---")

        x_next = generate_candidates(1)[0]

        y_next, Yexp_next = run_experiment(x_next)

        X = np.vstack((X, x_next))
        y = np.append(y, y_next)

        Y_full = np.concatenate(
            (Y_full, Yexp_next),
            axis=2
        )

        params = estimate_parameters(
            X,
            Y_full,
            initial_guess=params
        )

        param_history.append(params)

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
# PARAMETER ERROR
# ============================================================

def parameter_error(param_history):

    err = []

    for p in param_history:

        rel_err = (
            np.linalg.norm(p - TRUE_K)
            / np.linalg.norm(TRUE_K)
        )

        err.append(100 * rel_err)

    return np.array(err)



# ============================================================
# BOED vs RANDOM COMPARISON
# ============================================================

def compare_boed_vs_random(
    param_hist_boed,
    param_hist_random
):

    err_boed = parameter_error(param_hist_boed)
    err_rand = parameter_error(param_hist_random)

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1,len(err_boed)+1),
        err_boed,
        marker='o',
        linewidth=2,
        label='BOED'
    )

    plt.plot(
        range(1,len(err_rand)+1),
        err_rand,
        marker='s',
        linewidth=2,
        label='Random'
    )

    plt.xlabel("Experiment Number")
    plt.ylabel("Parameter Error (%)")

    plt.title("BOED vs Random Design")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":

    # ==========================================
    # BOED
    # ==========================================

    X_boed, y_boed, Y_boed, param_boed = (
        bayesian_optimization(
            N_init=2,
            max_experiments=12,
            tol=1e-3
        )
    )

    summarize_results(
        X_boed,
        y_boed,
        param_boed
    )

    # ==========================================
    # RANDOM
    # ==========================================

    X_rand, y_rand, Y_rand, param_rand = (
        random_design(
            N_init=2,
            max_experiments=12
        )
    )

    summarize_results(
        X_rand,
        y_rand,
        param_rand
    )

    # ==========================================
    # BOED Plots
    # ==========================================

    plot_experiments(X_boed)

    plot_parameter_convergence(param_boed)

    plot_target_species(y_boed)

    # ==========================================
    # Comparison Plot
    # ==========================================

    compare_boed_vs_random(
        param_boed,
        param_rand
    )
    
    
    
    
    