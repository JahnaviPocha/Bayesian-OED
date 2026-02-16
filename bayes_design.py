# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:13:28 2026

@author: jahna
"""
# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, qmc

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Import necessary functions from the base CSTR code
from cstr import random_points_generator, ca_exp, parameter_estimator

# ============================================================
# STEP 2 — DEFINE EXPERIMENTAL DESIGN SPACE
# ============================================================

# Bounds for the design variables
CA_IN_BOUNDS = (0.1, 1.0)      # inlet concentration
TEMP_BOUNDS = (200.0, 400.0)   # temperature (K)

# Dictionary for convenience
DESIGN_BOUNDS = {
    "ca_in": CA_IN_BOUNDS,
    "Temp": TEMP_BOUNDS
}

# ============================================================
# STEP 3 — INITIAL EXPERIMENTS GENERATION
# ============================================================

def generate_initial_experiments(N_init=3):
    """
    Generate a small initial set of experiments using quasi-random sampling.
    """
    ca_in = random_points_generator(
        N_init,
        lb=(CA_IN_BOUNDS[0],),
        ub=(CA_IN_BOUNDS[1],)
    )

    Temp = random_points_generator(
        N_init,
        lb=(TEMP_BOUNDS[0],),
        ub=(TEMP_BOUNDS[1],)
    )

    return ca_in, Temp

def run_initial_experiments(N_init=3, true_k=(1.0, 20000.0),
                            add_noise=False, sigma=1e-6, order=1):
    """
    Run initial experiments and estimate parameters using the base CSTR code.
    """
    ca_in, Temp = generate_initial_experiments(N_init)

    ca_out = ca_exp(
        ca_ins=ca_in,
        Temp=Temp,
        Nexps=N_init,
        k=true_k,
        add_noise=add_noise,
        sigma=sigma,
        order=order
    )

    print("\nParameter estimation using initial experiments:")
    params = parameter_estimator(
        Nexps=N_init,
        ca_exp=ca_out,
        ca_in=ca_in,
        Temp=Temp,
        order=order,
        initial_guess=(0.1, 10.0)
    )

    history = {
        "Nexps": N_init,
        "ca_in": list(ca_in),
        "Temp": list(Temp),
        "ca_out": list(ca_out),
        "parameters": params
    }

    return history

# ============================================================
# STEP 4 — GAUSSIAN PROCESS SURROGATE MODEL
# ============================================================

def build_gp_model(X, y):
    """
    Build and train a Gaussian Process surrogate model.
    
    Parameters
    ----------
    X : ndarray of shape (N_samples, 2)
        Experimental inputs [ca_in, Temp]
    y : ndarray of shape (N_samples,)
        Corresponding outputs ca_out
    
    Returns
    -------
    gp : trained GaussianProcessRegressor
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.2, 20.0], length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=5
    )
    gp.fit(X, y)
    return gp

# ============================================================
# STEP 5 — EXPECTED IMPROVEMENT ACQUISITION FUNCTION
# ============================================================

def expected_improvement(X_candidate, gp, y_best, xi=0.01):
    """
    Compute the Expected Improvement (EI) at candidate points.
    
    Parameters
    ----------
    X_candidate : ndarray, shape (N_candidates, 2)
        Candidate experimental points
    gp : trained GaussianProcessRegressor
    y_best : float
        Best observed ca_out (to minimize)
    xi : float
        Exploration-exploitation trade-off
    
    Returns
    -------
    ei : ndarray, shape (N_candidates,)
        Expected Improvement at each candidate
    """
    mu, sigma = gp.predict(X_candidate, return_std=True)
    sigma = sigma.reshape(-1)

    with np.errstate(divide='warn'):
        improvement = y_best - mu - xi
        Z = np.zeros_like(improvement)
        ei = np.zeros_like(improvement)
        mask = sigma > 0
        Z[mask] = improvement[mask] / sigma[mask]
        ei[mask] = improvement[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
        ei[~mask] = 0.0

    return ei

# ============================================================
# STEP 6 — SELECT NEXT EXPERIMENT
# ============================================================

def select_next_experiment(gp, X_existing, y_existing, n_candidates=1000):
    """
    Propose the next experimental point using Expected Improvement.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Trained GP surrogate
    X_existing : ndarray
        Already performed experiments
    y_existing : ndarray
        Corresponding outputs
    n_candidates : int
        Number of random candidate points
    
    Returns
    -------
    x_next : ndarray of shape (2,)
        Next experiment [ca_in, Temp]
    """
    ca_candidates = np.random.uniform(CA_IN_BOUNDS[0], CA_IN_BOUNDS[1], n_candidates)
    T_candidates = np.random.uniform(TEMP_BOUNDS[0], TEMP_BOUNDS[1], n_candidates)
    X_candidates = np.column_stack((ca_candidates, T_candidates))

    y_best = np.min(y_existing)
    ei = expected_improvement(X_candidates, gp, y_best)

    x_next = X_candidates[np.argmax(ei)]
    return x_next

# ============================================================
# STEP 7 — STOPPING CRITERION
# ============================================================

def check_stopping(parameter_history, tol=1e-4):
    """
    Stops BO if parameter estimates stop changing.
    
    Parameters
    ----------
    parameter_history : list of ndarray
        Parameter estimates at each iteration
    tol : float
        Convergence tolerance
    
    Returns
    -------
    stop : bool
        True if change is below tol
    """
    if len(parameter_history) < 2:
        return False

    delta = np.linalg.norm(parameter_history[-1] - parameter_history[-2])
    print(f"Parameter change norm: {delta:.3e}")
    return delta < tol

# ============================================================
# STEP 8 — SEQUENTIAL BAYESIAN OPTIMIZATION LOOP
# ============================================================

def sequential_bo_loop(
    X_init,
    y_init,
    max_experiments=10,
    param_tol=1e-4,
    true_k=(1.0, 20000.0),
    add_noise=False,
    sigma=1e-6,
    N_repeats=5
):
    """
    Run the sequential Bayesian Optimization loop.
    
    Parameters
    ----------
    X_init : ndarray, shape (N0, 2)
        Initial experimental points [ca_in, Temp]
    y_init : ndarray, shape (N0,)
        Corresponding outputs ca_out
    max_experiments : int
        Maximum total experiments including initial
    param_tol : float
        Convergence tolerance for parameter change
    true_k : tuple
        True reaction parameters for synthetic experiments
    add_noise : bool
        Whether to add noise in synthetic experiments
    sigma : float
        Noise standard deviation
    N_repeats : int
        Replicates per experiment
    
    Returns
    -------
    X : ndarray
        All experimental points
    y : ndarray
        Observed ca_out
    parameter_history : ndarray
        Parameter estimates at each iteration
    """
    X = X_init.copy()
    y = y_init.copy()
    parameter_history = []

    print("\nStarting Bayesian Optimization Loop\n")

    for iteration in range(max_experiments - len(X_init)):
        print(f"\n--- BO Iteration {iteration + 1} ---")

        # STEP 1: Build GP surrogate
        gp = build_gp_model(X, y)

        # STEP 2: Select next experiment using EI
        x_next = select_next_experiment(gp, X, y)
        ca_next, T_next = x_next

        # STEP 3: Perform virtual experiment using CSTR model
        y_next = ca_exp(
            ca_ins=np.array([ca_next]),
            Temp=np.array([T_next]),
            Nexps=1,
            k=true_k,
            add_noise=add_noise,
            sigma=sigma,
            N_repeats=N_repeats
        )[0]

        # Append new data
        X = np.vstack((X, x_next))
        y = np.append(y, y_next)

        print(f"Selected experiment: ca_in={ca_next:.3f}, Temp={T_next:.1f}")
        print(f"Observed ca_out={y_next:.6f}")

        # STEP 4: Parameter estimation using current dataset
        params = parameter_estimator(
            Nexps=len(X),
            ca_exp=y,
            ca_in=X[:, 0],
            Temp=X[:, 1],
            initial_guess=(0.5, 15000.0)
        )
        parameter_history.append(params)

        # STEP 5: Check convergence
        if check_stopping(parameter_history, tol=param_tol):
            print("\nConvergence reached. Stopping BO.")
            break

    return X, y, np.array(parameter_history)


# ============================================================
# STEP 11 — FINAL OUTPUT & SUMMARY
# ============================================================

def summarize_results(X, y, parameter_history):
    """
    Print a summary of Bayesian optimization results.
    
    Parameters
    ----------
    X : ndarray
        Experimental points [ca_in, Temp]
    y : ndarray
        Observed outlet concentration
    parameter_history : ndarray
        Parameter estimates at each iteration
    
    Returns
    -------
    final_params : ndarray
        Final estimated parameters
    """
    print("\n===== FINAL OPTIMIZATION SUMMARY =====")
    print(f"Total experiments used: {len(X)}")

    final_params = parameter_history[-1]
    print(f"Final estimated parameters:")
    print(f"  k0 = {final_params[0]:.6f}")
    print(f"  Ea = {final_params[1]:.2f}")

    print("\nFinal experimental points:")
    for i, (ca, T, cout) in enumerate(zip(X[:, 0], X[:, 1], y)):
        print(f"Exp {i+1}: ca_in={ca:.3f}, Temp={T:.1f}, ca_out={cout:.6f}")

    return final_params

# ============================================================
# STEP 12 — DATA LOGGING
# ============================================================

def log_data(X, y, parameter_history):
    """
    Store the results in a dictionary for further use.
    """
    log = {
        "X": X,
        "y": y,
        "parameter_history": parameter_history
    }
    return log

# ============================================================
# STEP 13 — PLOTTING FUNCTIONS
# ============================================================

# 13.1 Experimental Design Plot
def plot_experiments(X):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:, 0], X[:, 1], c=np.arange(len(X)), s=80, cmap='viridis')
    plt.colorbar(label="Experiment order")
    plt.xlabel("ca_in")
    plt.ylabel("Temperature (K)")
    plt.title("Bayesian-Optimized Experimental Points")
    plt.grid(True)
    plt.show()

# 13.2 Parameter Convergence Plots
def plot_parameter_convergence(parameter_history):
    params = np.array(parameter_history)
    exp_numbers = np.arange(1, len(params) + 1)

    # k0 convergence
    plt.figure(figsize=(6,4))
    plt.plot(exp_numbers, params[:, 0], marker='o')
    plt.xlabel("Experiment number")
    plt.ylabel("Estimated k₀")
    plt.title("Convergence of Pre-exponential Factor (k₀)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Ea convergence
    plt.figure(figsize=(6,4))
    plt.plot(exp_numbers, params[:, 1], marker='s', color='orange')
    plt.xlabel("Experiment number")
    plt.ylabel("Estimated Activation Energy (Eₐ)")
    plt.title("Convergence of Activation Energy (Eₐ)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 13.3 Output Response Plot
def plot_ca_out(y):
    plt.figure(figsize=(6,4))
    plt.plot(y, marker='o', linestyle='-', color='green')
    plt.xlabel("Experiment number")
    plt.ylabel("Outlet concentration (ca_out)")
    plt.title("Observed Outlet Concentration")
    plt.grid(True)
    plt.show()

# ============================================================
# STEP 14 — MAIN DRIVER FOR FINAL RESULTS
# ============================================================

if __name__ == "__main__":

    # Example: Using initial experiments
    X_init = np.array([
        [0.1, 200.0],
        [0.9, 400.0]
    ])

    y_init = ca_exp(
        X_init[:, 0],
        X_init[:, 1],
        Nexps=2,
        k=(1.0, 20000.0),
        add_noise=True,
        sigma=1e-4,
        N_repeats=5
    )

    # Run the sequential Bayesian Optimization Loop
    X_final, y_final, param_hist = sequential_bo_loop(
        X_init,
        y_init,
        max_experiments=8,
        param_tol=1e-5,
        true_k=(1.0, 20000.0),
        add_noise=True,
        sigma=1e-4,
        N_repeats=5
    )

    # Step 11: Summary
    final_params = summarize_results(X_final, y_final, param_hist)

    # Step 12: Log data
    results_log = log_data(X_final, y_final, param_hist)

    # Step 13: Plots
    plot_experiments(X_final)
    plot_parameter_convergence(param_hist)
    plot_ca_out(y_final)


