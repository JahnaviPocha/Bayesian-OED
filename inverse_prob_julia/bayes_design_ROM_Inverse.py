# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:02:33 2026

@author: jahna
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


# ============================================================
# JULIA INTERFACE
# ============================================================

from call_to_KPE_code import (  # noqa: E402
    experiments,
    parameter_estimator,
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3
TARGET_SPECIES_INDEX = 2

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

TRUE_K = np.array([4000.0, 4000.0])
INITIAL_GUESS = np.array([1000.0, 1000.0])

P_TOTAL = 50
RATIO = 0.1
N_REPEATS_BASE = 10

# High noise needs either more experiments or more repeated measurements.
# Otherwise the averaged data still contains too much noise for a reliable
# two-parameter inverse problem.
N_REPEATS_BY_NOISE = {
    1e-3: 100,
    1e-4: 50,
    1e-5: 20,
}

NOISE_LEVELS = [1e-3, 1e-4, 1e-5]

N_INIT = 4
MAX_EXPERIMENTS = 12
N_CANDIDATES = 200
CONVERGENCE_TOL = 1e-3

# For a fair noise comparison, keep all noise levels on the same budget.
# Set this to True only when you want each individual run to stop as soon as
# consecutive parameter estimates barely change.
ALLOW_EARLY_STOP_IN_SWEEP = False

# Reproducibility for Python-side designs and candidate clouds.
BASE_SEED = 12345

# y is the mean of repeated noisy observations, so the GP observation
# variance is approximately noise_level**2 / n_repeats.
GP_ALPHA_FLOOR = 1e-14

# The current acquisition maximizes target output, not a Fisher information
# criterion. This keeps your original objective but makes it more stable.
EI_XI = 0.01


def repeats_for_noise(noise_level):
    """Return the number of repeated measurements for a noise level."""
    for noise, repeats in N_REPEATS_BY_NOISE.items():
        if np.isclose(noise_level, noise):
            return repeats

    return N_REPEATS_BASE


# ============================================================
# DESIGN HELPERS
# ============================================================

def complete_Y_in(Y_partial):
    """Convert independent mass fractions to the full inlet vector."""
    last = 1.0 - np.sum(Y_partial)

    if last < 0.0:
        return None

    return np.append(Y_partial, last)


def decode_design_vector(x):
    """Decode x = [Y1, Y2, ..., Y(n-1), Temp]."""
    x = np.asarray(x, dtype=float)
    Y_partial = x[:-1]
    Temp = float(x[-1])

    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, Temp


def validate_yexp_shape(
    Yexp,
    nexps_expected,
    name="Yexp",
    n_repeats=N_REPEATS_BASE,
):
    """Check Julia experiment tensor shape: (N_repeats, Nspec, Nexps)."""
    arr = np.asarray(Yexp, dtype=float)
    expected = (n_repeats, NSPEC, nexps_expected)

    if arr.shape != expected:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected}. "
            "Julia experiments() should return (N_repeats, Nspec, Nexps)."
        )

    return arr
# ============================================================
# EXPERIMENT + PARAMETER ESTIMATION WRAPPERS
# ============================================================

def run_experiment(x, noise_level, n_repeats):
    """
    Run one CFD experiment through Julia.

    Returns
    -------
    y_scalar:
        Mean target-species output used by the GP.
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
        N_repeats=n_repeats,
        std_data=noise_level,
        Nspec=NSPEC,
        k_true=TRUE_K.tolist(),
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1, n_repeats=n_repeats)

    # Average over repeated noisy measurements, then select target species.
    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    y_scalar = float(Y_mean[TARGET_SPECIES_INDEX])

    return y_scalar, Yexp


def estimate_parameters(
    X,
    Y_outputs,
    noise_level,
    initial_guess=None,
    n_repeats=N_REPEATS_BASE,
):
    """Call Julia parameter_estimator with explicit shape checks."""
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
    Y_out = validate_yexp_shape(
        Y_outputs,
        nexps_expected=Nexps,
        name="Y_outputs",
        n_repeats=n_repeats,
    )

    if initial_guess is None:
        initial_guess = INITIAL_GUESS

    initial_guess = np.clip(np.asarray(initial_guess, dtype=float), 0.0, 20000.0)

    print("\nDEBUG:")
    print("Y_in:", Y_in_all.shape)
    print("Temp:", Temp_all.shape)
    print("Y_out:", Y_out.shape)

    # Use a dict so the non-ASCII Julia keyword can stay ASCII in this file.
    kwargs = {
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_TOTAL,
        "St": np.array([[-2.0, -1.0, 2.0]]),
        "nref": 2500,
        "nreac": 1,
        "Nexps": Nexps,
        "Y_out": Y_out,
        "unknown_parameters": 2,
        "IG": initial_guess,
        "N_repeats": n_repeats,
        "\u03c3_data": noise_level,
        "RBS_full": False,
    }

    params = parameter_estimator(**kwargs)
    params = np.asarray(params, dtype=float).reshape(-1)

    print("params:", params)

    return params


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design(N_init=3, rng=None):
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


def generate_candidates(n_candidates=500, rng=None):
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


def gp_alpha_from_noise(noise_level, n_repeats=N_REPEATS_BASE):
    """Observation variance for the GP target y_mean."""
    return max((noise_level ** 2) / max(n_repeats, 1), GP_ALPHA_FLOOR)


def build_gp_model(X, y, noise_level, n_repeats):
    """Train a Gaussian Process surrogate."""
    X_scaled = scale_X(X)
    alpha = gp_alpha_from_noise(noise_level, n_repeats)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=BASE_SEED,
    )

    gp.fit(X_scaled, y)

    return gp


def expected_improvement(X_candidates, gp, y_best, xi=EI_XI):
    """Expected improvement for maximizing target output."""
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


def select_next_experiment(gp, X, y, rng=None, n_candidates=500):
    """Choose the next experiment using expected improvement."""
    X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
    y_best = np.max(y)
    ei = expected_improvement(X_candidates, gp, y_best)
    best_idx = int(np.argmax(ei))

    return X_candidates[best_idx]

# ============================================================
# BO LOOP
# ============================================================

def check_parameter_convergence(param_history, tol=1e-3):
    """Check relative change in consecutive k estimates."""
    if len(param_history) < 2:
        return False

    prev = np.asarray(param_history[-2], dtype=float)
    curr = np.asarray(param_history[-1], dtype=float)
    denom = np.maximum(np.abs(prev), np.abs(TRUE_K))
    denom = np.maximum(denom, 1e-30)
    delta = np.linalg.norm((curr - prev) / denom) / np.sqrt(curr.size)

    print(f"Relative parameter change: {delta:.4e}")

    return delta < tol


def bayesian_optimization(
    noise_level,
    N_init=3,
    max_experiments=10,
    tol=1e-3,
    n_candidates=500,
    allow_early_stop=True,
    rng_seed=None,
    initial_design=None,
):
    """
    Full BOED loop.

    For a multi-noise comparison, pass the same initial_design and rng_seed to
    each noise level, and set allow_early_stop=False.
    """
    print("\n=== STARTING BAYESIAN OPTIMIZATION ===\n")
    n_repeats = repeats_for_noise(noise_level)
    print(f"Noise level: {noise_level:.0e}")
    print(f"N_repeats: {n_repeats}")
    print(f"GP alpha: {gp_alpha_from_noise(noise_level, n_repeats):.3e}")

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
        y_scalar, Yexp = run_experiment(x, noise_level, n_repeats)
        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)
        print(f"Init Exp {i + 1}: y = {y_scalar:.6f}")

    y = np.asarray(y_list, dtype=float)
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=N_init,
        name="Y_full",
        n_repeats=n_repeats,
    )

    print("\nInitial parameter estimation...\n")

    params = estimate_parameters(
        X,
        Y_full,
        noise_level,
        initial_guess=INITIAL_GUESS,
        n_repeats=n_repeats,
    )
    param_history = [params]
    param_exp_counts = [len(X)]

    print(f"Initial parameters: {params}")

    for iteration in range(max_experiments - N_init):
        print(f"\n--- BO Iteration {iteration + 1} ---")

        gp = build_gp_model(X, y, noise_level, n_repeats)
        x_next = select_next_experiment(
            gp,
            X,
            y,
            rng=rng,
            n_candidates=n_candidates,
        )

        y_next, Yexp_next = run_experiment(x_next, noise_level, n_repeats)

        print("New experiment:")
        print(f"  x = {x_next}")
        print(f"  target output = {y_next:.6f}")

        X = np.vstack((X, x_next))
        y = np.append(y, y_next)
        Y_full = validate_yexp_shape(
            np.concatenate((Y_full, Yexp_next), axis=2),
            nexps_expected=len(X),
            name="Y_full",
            n_repeats=n_repeats,
        )

        # Warm-start the nonlinear inverse problem from the previous estimate.
        # This is much more stable for high-noise runs than restarting from
        # [1000, 1000] after every new experiment.
        params = estimate_parameters(
            X,
            Y_full,
            noise_level,
            initial_guess=param_history[-1],
            n_repeats=n_repeats,
        )
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
        n_repeats,
    )


# ============================================================
# SUMMARY
# ============================================================

def noise_label(noise):
    return f"{noise:.0e}"


def summarize_results(noise_level, X, y, param_history):
    print("\n===== FINAL RESULTS SUMMARY =====\n")
    print(f"Noise level: {noise_level:.0e}")
    print(f"Total experiments used: {len(X)}")

    final_params = np.asarray(param_history[-1], dtype=float)
    final_error = final_params - TRUE_K

    print("\nFinal estimated parameters:")
    print(f"  k1 = {final_params[0]:.6f}  error = {final_error[0]:+.6e}")
    print(f"  k2 = {final_params[1]:.6f}  error = {final_error[1]:+.6e}")

    best_idx = int(np.argmax(y))

    print("\nBest experiment by target output:")
    print(f"  X = {X[best_idx]}")
    print(f"  target output = {y[best_idx]:.6f}")

    return final_params


def print_noise_sweep_table(all_results):
    print("\n===== NOISE SWEEP SUMMARY =====\n")
    print("noise      repeats  n_exp    k1_final       k2_final       |k-true|")
    print("---------------------------------------------------------------------")

    for result in all_results:
        params = result["final_params"]
        err_norm = np.linalg.norm(params - TRUE_K)
        print(
            f"{result['noise']:.0e}   "
            f"{result['n_repeats']:7d}   "
            f"{len(result['X']):5d}   "
            f"{params[0]:12.6f}   "
            f"{params[1]:12.6f}   "
            f"{err_norm:10.3e}"
        )


# ============================================================
# COMBINED PLOTS
# ============================================================

def apply_plain_k_axis(ax):
    """Disable Matplotlib +4e3 style offset on k axes."""
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_all_experiments(all_results):
    plt.figure(figsize=(8, 5))

    line_styles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    alphas = [1.0, 0.8, 0.6]

    for i, result in enumerate(all_results):
        X = result["X"]

        plt.plot(
            X[:, 0],
            X[:, -1],
            linestyle=line_styles[i],
            marker=markers[i],
            linewidth=2,
            markersize=6,
            alpha=alphas[i],
            label=f"noise {noise_label(result['noise'])}"
        )

    plt.xlabel("Y1 (species 1)")
    plt.ylabel("Temperature (K)")
    plt.title("Experimental Designs Across Noise Levels")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_parameter_convergence(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        label = f"noise {noise_label(result['noise'])}"

        axes[0].plot(exp_counts, params[:, 0], marker="o", label=label)
        axes[1].plot(exp_counts, params[:, 1], marker="s", label=label)

    for i, ax in enumerate(axes):
        ax.axhline(TRUE_K[i], color="black", linestyle="--", linewidth=1.5)
        ax.axhline(INITIAL_GUESS[i], color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"k{i + 1}")
        ax.set_title(f"Parameter Convergence: k{i + 1}")
        ax.grid(True)
        apply_plain_k_axis(ax)

    axes[0].legend()
    fig.suptitle("Parameter Convergence Across Noise Levels")
    fig.tight_layout()
    plt.show()


def plot_all_parameter_error(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        label = f"noise {noise_label(result['noise'])}"

        axes[0].plot(
            exp_counts,
            (params[:, 0] - TRUE_K[0]) / TRUE_K[0],
            marker="o",
            label=label,
        )
        axes[1].plot(
            exp_counts,
            (params[:, 1] - TRUE_K[1]) / TRUE_K[1],
            marker="s",
            label=label,
        )

    for i, ax in enumerate(axes):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"relative error in k{i + 1}")
        ax.set_title(f"Relative Parameter Error: k{i + 1}")
        ax.grid(True)
        ax.set_yscale("symlog", linthresh=1e-4)

    axes[0].legend()
    fig.suptitle("Parameter Error Across Noise Levels")
    fig.tight_layout()
    plt.show()


def plot_all_outputs(all_results):
    plt.figure(figsize=(8, 5))

    for result in all_results:
        y = np.asarray(result["y"], dtype=float)
        exp_numbers = np.arange(1, len(y) + 1)
        label = f"noise {noise_label(result['noise'])}"
        plt.plot(exp_numbers, y, marker="o", linewidth=1.5, label=label)

    plt.xlabel("Experiment number")
    plt.ylabel("Target output fraction")
    plt.title("Target Output Across Experiments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_noise_sweep(all_results):
    """Create all combined comparison plots after the sweep is complete."""
    plot_all_experiments(all_results)
    plot_all_parameter_convergence(all_results)
    plot_all_parameter_error(all_results)
    plot_all_outputs(all_results)


# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":
    all_results = []

    # Same initial design for all noise levels.
    initial_rng = np.random.default_rng(BASE_SEED)
    shared_initial_design = generate_initial_design(N_INIT, rng=initial_rng)

    for noise in NOISE_LEVELS:
        print("\n")
        print("=" * 60)
        print(f"RUNNING BOED FOR NOISE = {noise:.0e}")
        print("=" * 60)

        (
            X,
            y,
            Y_full,
            param_history,
            param_exp_counts,
            n_repeats,
        ) = bayesian_optimization(
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
                "n_repeats": n_repeats,
            }
        )

    print_noise_sweep_table(all_results)
    plot_noise_sweep(all_results)
