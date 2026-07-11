# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:53:23 2026

@author: jahna
"""

"""
Bayesian OED driver for the methanol-production Julia model.

This follows the corrected ROM BOED structure:
- one Python driver loops over noise levels
- Julia experiments() supplies noisy outlet data
- the GP uses the mean methanol output as the acquisition objective
- Julia parameter_estimator() receives the full Y tensor
- all noise levels are plotted together after the sweep

Expected Julia tensor shape:
    Y_out = (N_repeats, Nspec, Nexps)

Put this file in:
    inverse_prob_julia/

and put call_to_KPE_code_meoh.py in the same folder.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

from call_to_KPE_code_meoh import experiments, parameter_estimator


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 6
N_REACTIONS = 2
N_UNKNOWN_PARAMETERS = 9

# Species order from main_meoh.jl molar_weights:
# CO2, H2, H2O, CH3OH, CO, N2
TARGET_SPECIES_INDEX = 3  # CH3OH, Python 0-based index

Y_BOUNDS = [
    (0.10, 0.33),  # CO2
    (0.10, 0.25),  # H2
    (0.00, 0.01),  # H2O
    (0.00, 0.30),  # CH3OH
    (0.00, 0.01),  # CO
]
TEMP_BOUNDS = (450.0, 550.0)

P_TOTAL = 50
RATIO = 0.1
N_REPEATS = 10

ST = np.array(
    [
        [-1.0, -3.0, 1.0, 1.0, 0.0, 0.0],
        [-1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    ]
)

# Physical methanol parameters from main_meoh.jl.
TRUE_PARAMS = np.array(
    [1.07, 3453.38, 0.499, 6.62e-11, 1.22e10, 40000.0, 17197.0, 124119.0, -98084.0],
    dtype=float,
)

PARAM_LOWER = np.array(
    [1e-2, 1e0, 1e-4, 1e-12, 1e7, 1e1, 1e1, 1e2, -2e5],
    dtype=float,
)
PARAM_UPPER = np.array(
    [1e2, 1e4, 1e2, 1e-8, 1e11, 1e5, 1e5, 1e6, -2e1],
    dtype=float,
)

TRUE_PARAMS_NORMALIZED = (TRUE_PARAMS - PARAM_LOWER) / (PARAM_UPPER - PARAM_LOWER)

# Matches main_meoh.jl complete_workflow().
INITIAL_GUESS_NORMALIZED = 0.5 * TRUE_PARAMS_NORMALIZED
INITIAL_GUESS_PHYSICAL = (
    INITIAL_GUESS_NORMALIZED * (PARAM_UPPER - PARAM_LOWER) + PARAM_LOWER
)

NOISE_LEVELS = [1e-3, 1e-5, 1e-7]

# Methanol has 9 unknown parameters, so more experiments are usually needed
# than the 2-parameter ROM case. These values are moderate for testing.
N_INIT = 3
MAX_EXPERIMENTS = 15
N_CANDIDATES = 50

# Keep False for fair noise-comparison plots.
ALLOW_EARLY_STOP_IN_SWEEP = False
CONVERGENCE_TOL = 1e-3

BASE_SEED = 12345
GP_ALPHA_FLOOR = 1e-16
EI_XI = 0.01

# RBS_full=true appears incomplete in main_meoh.jl because rbs_snapshot is not
# filled before being passed to newton_optimizer. main_meoh complete_workflow()
# uses RBS_full=false, so that is the safe default here too.
RBS_FULL = False




# ============================================================
# DESIGN HELPERS
# ============================================================

def complete_Y_in(Y_partial):
    """Convert independent inlet mass fractions to full 6-species vector."""
    last = 1.0 - np.sum(Y_partial)

    if last < 0.0:
        return None

    return np.append(Y_partial, last)


def decode_design_vector(x):
    """Decode x = [Y1, Y2, Y3, Y4, Y5, Temp]."""
    x = np.asarray(x, dtype=float)
    Y_partial = x[:-1]
    Temp = float(x[-1])

    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, Temp


def validate_yexp_shape(Yexp, nexps_expected, name="Yexp"):
    """Check Julia experiment tensor shape: (N_repeats, Nspec, Nexps)."""
    arr = np.asarray(Yexp, dtype=float)
    expected = (N_REPEATS, NSPEC, nexps_expected)

    if arr.shape != expected:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected}. "
            "Julia experiments() should return (N_repeats, Nspec, Nexps)."
        )

    return arr


def normalize_parameter_vector(params):
    """Convert physical parameters to the normalized scale used by Julia."""
    return (np.asarray(params, dtype=float) - PARAM_LOWER) / (PARAM_UPPER - PARAM_LOWER)


def denormalize_parameter_vector(params_normalized):
    """Convert normalized parameters back to physical scale."""
    return np.asarray(params_normalized, dtype=float) * (PARAM_UPPER - PARAM_LOWER) + PARAM_LOWER


# ============================================================
# JULIA WRAPPERS
# ============================================================

def run_experiment(x, noise_level):
    """
    Run one methanol CFD experiment through Julia.

    Returns
    -------
    y_scalar:
        Mean CH3OH outlet fraction used by the GP.
    Yexp:
        Full noisy tensor with shape (N_repeats, Nspec, 1).
    """
    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        raise ValueError(f"Invalid design vector with negative final species: {x}")

    Y_in = np.asarray(Y_in, dtype=float).reshape(NSPEC, 1)
    Temp = np.asarray([Temp], dtype=float)

    Yexp = experiments(
        Y_in=Y_in,
        Temp=Temp,
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)

    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    y_scalar = float(Y_mean[TARGET_SPECIES_INDEX])

    return y_scalar, Yexp


def estimate_parameters(X, Y_outputs, noise_level):
    """Call Julia parameter_estimator for the methanol model."""
    X = np.asarray(X, dtype=float)
    Nexps = X.shape[0]

    Y_in_all = []
    Temp_all = []

    for x in X:
        Y_in, Temp = decode_design_vector(x)

        if Y_in is None:
            raise ValueError(f"Invalid design vector in X: {x}")

        Y_in_all.append(Y_in)
        Temp_all.append(Temp)

    Y_in_all = np.asarray(Y_in_all, dtype=float).T
    Temp_all = np.asarray(Temp_all, dtype=float)
    Y_out = validate_yexp_shape(Y_outputs, nexps_expected=Nexps, name="Y_outputs")

    print("\nDEBUG:")
    print("Y_in:", Y_in_all.shape)
    print("Temp:", Temp_all.shape)
    print("Y_out:", Y_out.shape)

    kwargs = {
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_TOTAL,
        "St": ST,
        "nref": 2500,
        "nreac": N_REACTIONS,
        "Nexps": Nexps,
        "Y_out": Y_out,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": INITIAL_GUESS_NORMALIZED.copy(),
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
    }

    params_physical = parameter_estimator(**kwargs)
    params_physical = np.asarray(params_physical, dtype=float).reshape(-1)

    print("params physical:", params_physical)

    return params_physical


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design(N_init=5, rng=None):
    """Generate valid initial BO points."""
    rng = np.random.default_rng() if rng is None else rng
    X_init = []

    while len(X_init) < N_init:
        Y_partial = [
            rng.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]
        Temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [Temp], dtype=float)

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            X_init.append(x)

    return np.asarray(X_init, dtype=float)


def generate_candidates(n_candidates=200, rng=None):
    """Random candidate sampling in the valid design space."""
    rng = np.random.default_rng() if rng is None else rng
    candidates = []

    while len(candidates) < n_candidates:
        Y_partial = [
            rng.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]
        Temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [Temp], dtype=float)

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            candidates.append(x)

    return np.asarray(candidates, dtype=float)


# ============================================================
# GP + EXPECTED IMPROVEMENT
# ============================================================

def scale_X(X):
    """Scale design variables to [0, 1] for GP stability."""
    X = np.asarray(X, dtype=float)
    X_scaled = np.zeros_like(X)

    for i in range(NSPEC - 1):
        lb, ub = Y_BOUNDS[i]
        X_scaled[:, i] = (X[:, i] - lb) / (ub - lb)

    lb, ub = TEMP_BOUNDS
    X_scaled[:, -1] = (X[:, -1] - lb) / (ub - lb)

    return X_scaled


def gp_alpha_from_noise(noise_level):
    """Observation variance for the GP target y_mean."""
    return max((noise_level ** 2) / max(N_REPEATS, 1), GP_ALPHA_FLOOR)


def build_gp_model(X, y, noise_level):
    """Train a Gaussian Process surrogate."""
    X_scaled = scale_X(X)
    alpha = gp_alpha_from_noise(noise_level)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=BASE_SEED,
    )

    gp.fit(X_scaled, y)

    return gp


def expected_improvement(X_candidates, gp, y_best, xi=EI_XI):
    """Expected improvement for maximizing methanol outlet fraction."""
    X_scaled = scale_X(X_candidates)

    mu, sigma = gp.predict(X_scaled, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    sigma_safe = np.maximum(sigma, 1e-15)

    improvement = mu - y_best - xi
    Z = improvement / sigma_safe
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0

    return ei.ravel()


def select_next_experiment(gp, X, y, rng=None, n_candidates=200):
    """Choose the next experiment using expected improvement."""
    X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
    y_best = np.max(y)
    ei = expected_improvement(X_candidates, gp, y_best)
    best_idx = int(np.argmax(ei))

    return X_candidates[best_idx]


# ============================================================
# BO LOOP
# ============================================================

def relative_parameter_change(param_history):
    """Relative norm change for physical parameters with very different scales."""
    if len(param_history) < 2:
        return np.inf

    prev = np.asarray(param_history[-2], dtype=float)
    curr = np.asarray(param_history[-1], dtype=float)
    denom = np.maximum(np.abs(prev), np.abs(TRUE_PARAMS))
    denom = np.maximum(denom, 1e-30)

    return np.linalg.norm((curr - prev) / denom) / np.sqrt(curr.size)


def check_parameter_convergence(param_history, tol=1e-3):
    """Stop if relative parameter estimates stop changing."""
    delta = relative_parameter_change(param_history)
    print(f"Relative parameter change: {delta:.4e}")

    return delta < tol

def bayesian_optimization(
    noise_level,
    N_init=5,
    max_experiments=10,
    tol=1e-3,
    n_candidates=200,
    allow_early_stop=True,
    rng_seed=None,
    initial_design=None,
):
    """
    Full BOED loop for methanol.

    For fair multi-noise comparison, pass the same initial_design and rng_seed
    to each noise level, and set allow_early_stop=False.
    """
    print("\n=== STARTING METHANOL BAYESIAN OED ===\n")
    print(f"Noise level: {noise_level:.0e}")
    print(f"GP alpha: {gp_alpha_from_noise(noise_level):.3e}")

    rng = np.random.default_rng(rng_seed)

    if initial_design is None:
        X = generate_initial_design(N_init, rng=rng)
    else:
        X = np.asarray(initial_design, dtype=float).copy()

        if X.shape[0] != N_init:
            raise ValueError(
                f"initial_design has {X.shape[0]} rows, expected N_init={N_init}."
            )

    y_list = []
    Y_tensor_list = []

    print("Running initial experiments...\n")

    for i, x in enumerate(X):
        y_scalar, Yexp = run_experiment(x, noise_level)
        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)
        print(f"Init Exp {i + 1}: methanol = {y_scalar:.6f}")

    y = np.asarray(y_list, dtype=float)
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=N_init,
        name="Y_full",
    )

    print("\nInitial parameter estimation...\n")

    params = estimate_parameters(X, Y_full, noise_level)
    param_history = [params]
    param_exp_counts = [len(X)]

    print(f"Initial parameters: {params}")

    for iteration in range(max_experiments - N_init):
        print(f"\n--- BO Iteration {iteration + 1} ---")

        gp = build_gp_model(X, y, noise_level)
        x_next = select_next_experiment(
            gp,
            X,
            y,
            rng=rng,
            n_candidates=n_candidates,
        )

        y_next, Yexp_next = run_experiment(x_next, noise_level)

        print("New experiment:")
        print(f"  x = {x_next}")
        print(f"  methanol = {y_next:.6f}")

        X = np.vstack((X, x_next))
        y = np.append(y, y_next)
        Y_full = validate_yexp_shape(
            np.concatenate((Y_full, Yexp_next), axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        params = estimate_parameters(X, Y_full, noise_level)
        param_history.append(params)
        param_exp_counts.append(len(X))

        print(f"Estimated parameters: {params}")

        if allow_early_stop and check_parameter_convergence(param_history, tol):
            print("\nConvergence reached -> stopping early.")
            break

    print("\n=== BO FINISHED ===\n")

    return (
        X,
        y,
        Y_full,
        np.asarray(param_history, dtype=float),
        np.asarray(param_exp_counts, dtype=int),
    )


# ============================================================
# SUMMARY
# ============================================================

def noise_label(noise):
    return f"{noise:.0e}"


def relative_error(params):
    params = np.asarray(params, dtype=float)
    denom = np.maximum(np.abs(TRUE_PARAMS), 1e-30)
    return (params - TRUE_PARAMS) / denom


def summarize_results(noise_level, X, y, param_history):
    print("\n===== FINAL RESULTS SUMMARY =====\n")
    print(f"Noise level: {noise_level:.0e}")
    print(f"Total experiments used: {len(X)}")

    final_params = np.asarray(param_history[-1], dtype=float)
    rel_err = relative_error(final_params)

    print("\nFinal estimated parameters:")
    for i, value in enumerate(final_params):
        print(
            f"  p{i + 1}: {value:.6e} "
            f"true={TRUE_PARAMS[i]:.6e} "
            f"rel_error={rel_err[i]:+.3e}"
        )

    best_idx = int(np.argmax(y))

    print("\nBest experiment by methanol outlet:")
    print(f"  X = {X[best_idx]}")
    print(f"  methanol = {y[best_idx]:.6f}")

    return final_params


def print_noise_sweep_table(all_results):
    print("\n===== NOISE SWEEP SUMMARY =====\n")
    print("noise      n_exp    methanol_best    rel_param_error_norm")
    print("----------------------------------------------------------")

    for result in all_results:
        params = result["final_params"]
        err_norm = np.linalg.norm(relative_error(params)) / np.sqrt(params.size)
        print(
            f"{result['noise']:.0e}   "
            f"{len(result['X']):5d}   "
            f"{np.max(result['y']):13.6f}   "
            f"{err_norm:20.3e}"
        )


# ============================================================
# COMBINED PLOTS
# ============================================================

def apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_all_experiments(all_results):
    plt.figure(figsize=(8, 5))

    for result in all_results:
        X = result["X"]
        label = f"noise {noise_label(result['noise'])}, n={len(X)}"
        plt.plot(X[:, 0], X[:, -1], marker="o", linewidth=1.5, label=label)

    plt.xlabel("CO2 inlet fraction")
    plt.ylabel("Temperature (K)")
    plt.title("Methanol BOED Designs Across Noise Levels")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_methanol_outputs(all_results):
    plt.figure(figsize=(8, 5))

    for result in all_results:
        y = np.asarray(result["y"], dtype=float)
        exp_numbers = np.arange(1, len(y) + 1)
        label = f"noise {noise_label(result['noise'])}"
        plt.plot(exp_numbers, y, marker="o", linewidth=1.5, label=label)

    plt.xlabel("Experiment number")
    plt.ylabel("Methanol outlet fraction")
    plt.title("Methanol Output Across Experiments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_relative_parameter_errors(all_results):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        errors = relative_error(params)
        label = f"noise {noise_label(result['noise'])}"

        for i, ax in enumerate(axes):
            ax.plot(exp_counts, errors[:, i], marker="o", linewidth=1.3, label=label)

    for i, ax in enumerate(axes):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"p{i + 1}")
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel("relative error")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True)

    axes[0].legend()
    fig.suptitle("Methanol Parameter Relative Error")
    fig.tight_layout()
    plt.show()


def plot_selected_physical_parameters(all_results, selected=(0, 1, 4, 8)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        label = f"noise {noise_label(result['noise'])}"

        for axis_index, param_index in enumerate(selected):
            axes[axis_index].plot(
                exp_counts,
                params[:, param_index],
                marker="o",
                linewidth=1.3,
                label=label,
            )

    for axis_index, param_index in enumerate(selected):
        ax = axes[axis_index]
        ax.axhline(TRUE_PARAMS[param_index], color="black", linestyle="--", linewidth=1.0)
        ax.axhline(
            INITIAL_GUESS_PHYSICAL[param_index],
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )
        ax.set_title(f"p{param_index + 1}")
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel("physical value")
        ax.grid(True)
        apply_plain_axis(ax)

    axes[0].legend()
    fig.suptitle("Selected Methanol Parameter Convergence")
    fig.tight_layout()
    plt.show()


def plot_noise_sweep(all_results):
    plot_all_experiments(all_results)
    plot_all_methanol_outputs(all_results)
    plot_relative_parameter_errors(all_results)
    plot_selected_physical_parameters(all_results)


# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":
    all_results = []

    initial_rng = np.random.default_rng(BASE_SEED)
    shared_initial_design = generate_initial_design(N_INIT, rng=initial_rng)

    for noise in NOISE_LEVELS:
        print("\n")
        print("=" * 60)
        print(f"RUNNING METHANOL BOED FOR NOISE = {noise:.0e}")
        print("=" * 60)

        X, y, Y_full, param_history, param_exp_counts = bayesian_optimization(
            noise_level=noise,
            N_init=N_INIT,
            max_experiments=MAX_EXPERIMENTS,
            tol=CONVERGENCE_TOL,
            n_candidates=N_CANDIDATES,
            allow_early_stop=ALLOW_EARLY_STOP_IN_SWEEP,
            rng_seed=BASE_SEED,
            initial_design=shared_initial_design,
        )

        final_params = summarize_results(noise, X, y, param_history)

        all_results.append(
            {
                "noise": noise,
                "X": X,
                "y": y,
                "Y_full": Y_full,
                "params": param_history,
                "param_exp_counts": param_exp_counts,
                "final_params": final_params,
            }
        )

    print_noise_sweep_table(all_results)
    plot_noise_sweep(all_results)
    
    
    
    
    