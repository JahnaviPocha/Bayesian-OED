# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:41:25 2026

@author: jahna
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    (0.00, 0.01),  # CH3OH
    (0.00, 0.30),  # CO
]
TEMP_BOUNDS = (450.0, 550.0)

PRESSURE_BOUNDS = (15.0, 50.0)
RATIO = 0.1
N_REPEATS = 10

# Physical methanol parameters from main_meoh.jl.
TRUE_PARAMS = np.array(
    [15672.02, 3453.38, 30.836, 558.532, 0.7439, 40000, 17197, 124119, 98084],
    dtype=float,
)

PARAMETER_NAMES = [
    "Reaction 1 pre-exponential factor A1",
    "Reaction 1 activation energy E1",
    "Reaction 1 adsorption constant K_ads,1",
    "Reaction 1 adsorption constant K_ads,2",
    "Reaction 2 pre-exponential factor A2",
    "Reaction 2 activation energy E2",
    "Reaction 2 adsorption constant K_ads,3",
    "Reaction 2 adsorption constant K_ads,4",
    "Global inhibition / adsorption parameter",
]

PARAMETER_SHORT_NAMES = [
    "A1",
    "E1",
    "K_ads,1",
    "K_ads,2",
    "A2",
    "E2",
    "K_ads,3",
    "K_ads,4",
    "Inhibition / adsorption",
]

DESIGN_AXIS_LABELS = [
    "CO2 inlet mass fraction",
    "H2 inlet mass fraction",
    "H2O inlet mass fraction",
    "CH3OH inlet mass fraction",
    "CO inlet mass fraction",
    "Temperature (K)",
    "Total pressure (bar)",
]

PARAM_LOWER = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1, 1e4, 1e4, 1e4, 1e4],
    dtype=float,
)

PARAM_UPPER = np.array(
    [1e5, 1e4, 1e5, 1e5, 1e5, 1.5e5, 1.5e5, 1.5e5, 1.5e5],
    dtype=float,
)

TRUE_PARAMS_NORMALIZED = (
    np.log10(TRUE_PARAMS) - np.log10(PARAM_LOWER)
) / (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))


def denormalize_parameter_vector(params_normalized):
    params_normalized = np.asarray(params_normalized, dtype=float)
    return 10 ** (
        params_normalized * (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))
        + np.log10(PARAM_LOWER)
    )

# Now this works:
INITIAL_GUESS_NORMALIZED = np.full(N_UNKNOWN_PARAMETERS, 0.1)

INITIAL_GUESS_PHYSICAL = denormalize_parameter_vector(INITIAL_GUESS_NORMALIZED)


NOISE_LEVELS = [1e-5, 1e-6, 1e-7]

# Methanol has 9 unknown parameters, so more experiments are usually needed
# than the 2-parameter ROM case. These values are moderate for testing.
N_INIT = 10
MAX_EXPERIMENTS = 100
N_CANDIDATES = 200

# Keep False for fair noise-comparison plots.
ALLOW_EARLY_STOP_IN_SWEEP = False
CONVERGENCE_TOL = 1e-3

BASE_SEED = 12345
GP_ALPHA_FLOOR = 1e-16

EI_XI = 0.01

# RBS_full=true appears incomplete in main_meoh.jl because rbs_snapshot is not
# filled before being passed to newton_optimizer. main_meoh complete_workflow()
# uses RBS_full=false, so that is the safe default here too.
RBS_FULL = True

# All plots, Excel data, and text summaries are saved under this folder.
PROJECT_DIR = Path(
    r"C:\Users\jahna\OneDrive\Desktop\masters\master's thesis"
    r"\Bayesian-OED\inverse_prob_julia"
)

RESULTS_FOLDER_NAME = "bayes_design_meoh_RBS"
ADD_TIMESTAMP_TO_RESULTS_FOLDER = False

folder_name = RESULTS_FOLDER_NAME
if ADD_TIMESTAMP_TO_RESULTS_FOLDER:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{RESULTS_FOLDER_NAME}_{stamp}"

RESULTS_DIR = PROJECT_DIR / "results" / folder_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = True
SHOW_PLOTS = True
PRINT_ESTIMATOR_SHAPES = True


# ============================================================
# OUTPUT HELPERS
# ============================================================

def noise_label(noise):
    return f"{noise:.0e}"


def safe_noise_label(noise):
    return noise_label(noise).replace("-", "m").replace("+", "p")


def save_and_show(fig, results_dir=None, filename=None):
    if SAVE_PLOTS and results_dir is not None and filename is not None:
        fig.savefig(results_dir / filename, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

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
    """Decode x = [Y1, Y2, Y3, Y4, Y5, Temp, Pressure]."""
    x = np.asarray(x, dtype=float)
    Y_partial = x[:-2]
    Temp = float(x[-2])
    Pressure = float(x[-1])

    Y_full = complete_Y_in(Y_partial)
    if Y_full is None:
        return None, None, None

    return Y_full, Temp, Pressure


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
    """Convert physical parameters to the normalized log-scale used by Julia."""
    params = np.asarray(params, dtype=float)
    return (np.log10(params) - np.log10(PARAM_LOWER)) / (
        np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER)
    )


# ============================================================
# JULIA WRAPPERS
# ============================================================

def run_experiment(x, noise_level):
    Y_in, Temp, Pressure = decode_design_vector(x)
    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Y_in = np.asarray(Y_in, dtype=float).reshape(NSPEC, 1)
    Temp = np.asarray([Temp], dtype=float)
    Pressure = np.asarray([Pressure], dtype=float)

    Yexp = experiments(
        Y_in=Y_in,
        Temp=Temp,
        P_total=Pressure,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
        scale=1.0,
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
    P_all = []

    for x in X:
        Y_in, Temp, Pressure = decode_design_vector(x)

        if Y_in is None:
            raise ValueError(f"Invalid design vector in X: {x}")

        Y_in_all.append(Y_in)
        Temp_all.append(Temp)
        P_all.append(Pressure)

    Y_in_all = np.asarray(Y_in_all, dtype=float).T
    Temp_all = np.asarray(Temp_all, dtype=float)
    P_all = np.asarray(P_all, dtype=float)
    Y_out = validate_yexp_shape(Y_outputs, nexps_expected=Nexps, name="Y_outputs")

    if PRINT_ESTIMATOR_SHAPES:
        print(
            "\nEstimator receives real design arrays: "
            f"Y_in={Y_in_all.shape}, "
            f"Temp={Temp_all.shape}, "
            f"P_total={P_all.shape}, "
            f"Y_out={Y_out.shape}"
        )

    kwargs = {
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_all,
        "St": np.array([
            [-1.0, -3.0, 1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 1.0, 0.0, 1.0, 0.0]
        ]),
        "nref": 2500,
        "nreac": N_REACTIONS,
        "Nexps": Nexps,
        "Y_out": Y_out,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": INITIAL_GUESS_NORMALIZED.copy(),
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
        "scale": 1.0,
    }

    params_physical = parameter_estimator(**kwargs)
    params_physical = np.asarray(params_physical, dtype=float).reshape(-1)

    print("Estimated physical parameters:", params_physical)

    return params_physical


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design( N_init=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X_init = []

    while len(X_init) < N_init:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        Temp = rng.uniform(*TEMP_BOUNDS)
        Pressure = rng.uniform(*PRESSURE_BOUNDS)

        x = np.asarray(Y_partial + [Temp, Pressure], dtype=float)
        Y_full, _, _ = decode_design_vector(x)

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
        Pressure = rng.uniform(*PRESSURE_BOUNDS)

        x = np.asarray(Y_partial + [Temp, Pressure], dtype=float)
        Y_full, _, _ = decode_design_vector(x)

        if Y_full is not None:
            candidates.append(x)

    return np.asarray(candidates, dtype=float)


# ============================================================
# GP + EXPECTED IMPROVEMENT
# ============================================================

def scale_X(X):
    """
    Scale design variables to [0, 1] for GP stability.

    These scaled values are used only inside the Python GP/EI model. They are
    never passed to Julia experiments() or parameter_estimator().
    """
    X = np.asarray(X, dtype=float)
    X_scaled = np.zeros_like(X)

    for i in range(NSPEC - 1):
        lb, ub = Y_BOUNDS[i]
        X_scaled[:, i] = (X[:, i] - lb) / (ub - lb)

    lb, ub = TEMP_BOUNDS
    X_scaled[:, -2] = (X[:, -2] - lb) / (ub - lb)

    lb, ub = PRESSURE_BOUNDS
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
    N_init=N_INIT,
    max_experiments=MAX_EXPERIMENTS,
    tol=1e-3,
    n_candidates=N_CANDIDATES,
    allow_early_stop=ALLOW_EARLY_STOP_IN_SWEEP,
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

    # BO iterations with top-2 batch per iteration
    for iteration in range(max_experiments - N_init):

        print(f"\n--- BO Iteration {iteration + 1} ---")

        gp = build_gp_model(X, y, noise_level)

        # Generate candidate batch
        X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)

        # Compute EI for all candidates
        y_best = np.max(y)
        ei = expected_improvement(X_candidates, gp, y_best)

        # Select top-2 candidates
        sorted_idx = np.argsort(ei)[::-1]
        top_k = 2
        X_batch = X_candidates[sorted_idx[:top_k]]

        print(f"Running {len(X_batch)} experiments in this BO iteration...")

        y_new_list = []
        Yexp_new_list = []

        for x_next in X_batch:
            y_next, Yexp_next = run_experiment(x_next, noise_level)
            print(f"  x={x_next}, methanol={y_next:.6f}")
            y_new_list.append(y_next)
            Yexp_new_list.append(Yexp_next)

        # Update data ONCE per iteration
        X = np.vstack((X, X_batch))
        y = np.append(y, y_new_list)
        Y_full = validate_yexp_shape(
            np.concatenate([Y_full] + Yexp_new_list, axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        # Parameter estimation (warm-start)
        params = estimate_parameters(X, Y_full, noise_level)
        param_history.append(params)
        param_exp_counts.append(len(X))

        print(f"Estimated parameters: {params}")

        # Stop if we reached max experiments
        if len(X) >= max_experiments:
            print("\nReached maximum experiments -> stopping.")
            break

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
# SUMMARY, EXPORT, AND PLOTTING
# ============================================================

NOISE_COLORS = {
    1e-5: "tab:blue",
    1e-6: "tab:orange",
    1e-7: "tab:green",
}

NOISE_MARKERS = {
    1e-5: "o",
    1e-6: "s",
    1e-7: "^",
}

NOISE_LINESTYLES = {
    1e-5: "-",
    1e-6: "--",
    1e-7: "-.",
}

SPECIES_NAMES = ["CO2", "H2", "H2O", "CH3OH", "CO", "N2"]


def style_for_noise(noise):
    return {
        "color": NOISE_COLORS.get(noise, None),
        "marker": NOISE_MARKERS.get(noise, "o"),
        "linestyle": NOISE_LINESTYLES.get(noise, "-"),
    }


def marker_size_for_noise(noise):
    return 11 if np.isclose(noise, 1e-5) else 7


def scatter_size_for_noise(noise):
    return 165 if np.isclose(noise, 1e-5) else 85


def display_offset_for_noise(noise, width=0.0):
    """
    Small display-only offset to reveal overlapping scatter points.

    The underlying X/y data are not changed. Use width=0 for line plots where
    exact x-axis values should be preserved.
    """
    try:
        idx = NOISE_LEVELS.index(noise)
    except ValueError:
        return 0.0

    center = 0.5 * (len(NOISE_LEVELS) - 1)
    return (idx - center) * width


def print_overlap_diagnostics(all_results):
    """Print whether different noise cases selected almost identical data."""
    print("\n===== PLOT OVERLAP DIAGNOSTICS =====")

    for i, first in enumerate(all_results):
        for second in all_results[i + 1:]:
            same_x = first["X"].shape == second["X"].shape and np.allclose(
                first["X"],
                second["X"],
                rtol=1e-10,
                atol=1e-12,
            )
            max_y_diff = np.max(np.abs(first["y"] - second["y"]))
            print(
                f"{noise_label(first['noise'])} vs {noise_label(second['noise'])}: "
                f"same X = {same_x}, max |methanol diff| = {max_y_diff:.3e}"
            )


def relative_error(params):
    params = np.asarray(params, dtype=float)
    denom = np.maximum(np.abs(TRUE_PARAMS), 1e-30)
    return (params - TRUE_PARAMS) / denom


def transformed_parameter_values(values, param_index):
    """
    Display transform requested for convergence plots.

    p1-p5 are shown as natural log values. p6-p9 are shown as log10 values.
    This affects plotting/export only; Julia still receives normalized IG.
    """
    values = np.asarray(values, dtype=float)

    if np.any(values <= 0.0):
        raise ValueError("Log transform requires positive parameter values.")

    if param_index < 5:
        return np.log(values), "ln(physical parameter value)", "Natural Log"

    return np.log10(values), "log10(physical parameter value)", "Log10"


def mixed_log_matrix(params):
    params = np.asarray(params, dtype=float)
    transformed = np.zeros_like(params, dtype=float)

    for param_index in range(N_UNKNOWN_PARAMETERS):
        transformed[:, param_index], _, _ = transformed_parameter_values(
            params[:, param_index],
            param_index,
        )

    return transformed


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


def build_summary_dataframe(all_results):
    rows = []

    for result in all_results:
        final_params = np.asarray(result["final_params"], dtype=float)
        rel_err = relative_error(final_params)
        best_idx = int(np.argmax(result["y"]))

        rows.append(
            {
                "noise": noise_label(result["noise"]),
                "std_data": result["noise"],
                "n_experiments": len(result["X"]),
                "best_experiment_number": best_idx + 1,
                "best_ch3oh_outlet_fraction": float(result["y"][best_idx]),
                "rms_relative_parameter_error": float(
                    np.linalg.norm(rel_err) / np.sqrt(rel_err.size)
                ),
                "max_abs_relative_parameter_error": float(np.max(np.abs(rel_err))),
            }
        )

    return pd.DataFrame(rows)


def build_final_parameters_dataframe(all_results):
    rows = []

    for result in all_results:
        final_params = np.asarray(result["final_params"], dtype=float)
        rel_err = relative_error(final_params)

        for param_index in range(N_UNKNOWN_PARAMETERS):
            final_transformed, ylabel, transform_name = transformed_parameter_values(
                [final_params[param_index]],
                param_index,
            )
            true_transformed, _, _ = transformed_parameter_values(
                [TRUE_PARAMS[param_index]],
                param_index,
            )

            rows.append(
                {
                    "noise": noise_label(result["noise"]),
                    "parameter_index": param_index + 1,
                    "short_name": PARAMETER_SHORT_NAMES[param_index],
                    "full_name": PARAMETER_NAMES[param_index],
                    "transform": transform_name,
                    "transform_axis_label": ylabel,
                    "true_physical": TRUE_PARAMS[param_index],
                    "initial_guess_physical": INITIAL_GUESS_PHYSICAL[param_index],
                    "final_estimate_physical": final_params[param_index],
                    "relative_error": rel_err[param_index],
                    "true_transformed": true_transformed[0],
                    "final_transformed": final_transformed[0],
                }
            )

    return pd.DataFrame(rows)


def build_parameter_history_dataframe(all_results):
    rows = []

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        rel_err = relative_error(params)

        for row_index, exp_count in enumerate(exp_counts):
            for param_index in range(N_UNKNOWN_PARAMETERS):
                estimate = params[row_index, param_index]
                transformed, ylabel, transform_name = transformed_parameter_values(
                    [estimate],
                    param_index,
                )

                rows.append(
                    {
                        "noise": noise_label(result["noise"]),
                        "total_experiments_used": int(exp_count),
                        "parameter_index": param_index + 1,
                        "short_name": PARAMETER_SHORT_NAMES[param_index],
                        "full_name": PARAMETER_NAMES[param_index],
                        "transform": transform_name,
                        "transform_axis_label": ylabel,
                        "estimate_physical": estimate,
                        "true_physical": TRUE_PARAMS[param_index],
                        "relative_error": rel_err[row_index, param_index],
                        "estimate_transformed": transformed[0],
                    }
                )

    return pd.DataFrame(rows)


def build_design_dataframe(all_results):
    rows = []

    for result in all_results:
        X = np.asarray(result["X"], dtype=float)
        y = np.asarray(result["y"], dtype=float)

        for exp_index, x in enumerate(X):
            Y_full, temp, pressure = decode_design_vector(x)
            row = {
                "noise": noise_label(result["noise"]),
                "experiment_number": exp_index + 1,
                "ch3oh_outlet_fraction": y[exp_index],
                "temperature_K": temp,
                "total_pressure_bar": pressure,
            }

            for species_index, species_name in enumerate(SPECIES_NAMES):
                row[f"{species_name}_inlet_mass_fraction"] = Y_full[species_index]

            rows.append(row)

    return pd.DataFrame(rows)


def build_y_output_dataframe(all_results):
    rows = []

    for result in all_results:
        Y_full = np.asarray(result["Y_full"], dtype=float)

        for repeat_index in range(Y_full.shape[0]):
            for species_index, species_name in enumerate(SPECIES_NAMES):
                for exp_index in range(Y_full.shape[2]):
                    rows.append(
                        {
                            "noise": noise_label(result["noise"]),
                            "repeat_number": repeat_index + 1,
                            "species_index": species_index + 1,
                            "species": species_name,
                            "experiment_number": exp_index + 1,
                            "outlet_mass_fraction": Y_full[
                                repeat_index,
                                species_index,
                                exp_index,
                            ],
                        }
                    )

    return pd.DataFrame(rows)


def build_settings_dataframe(results_dir):
    settings = {
        "results_dir": str(results_dir),
        "NSPEC": NSPEC,
        "N_REACTIONS": N_REACTIONS,
        "N_UNKNOWN_PARAMETERS": N_UNKNOWN_PARAMETERS,
        "TARGET_SPECIES_INDEX_python": TARGET_SPECIES_INDEX,
        "RATIO": RATIO,
        "N_REPEATS": N_REPEATS,
        "NOISE_LEVELS": ", ".join(noise_label(noise) for noise in NOISE_LEVELS),
        "N_INIT": N_INIT,
        "MAX_EXPERIMENTS": MAX_EXPERIMENTS,
        "N_CANDIDATES": N_CANDIDATES,
        "ALLOW_EARLY_STOP_IN_SWEEP": ALLOW_EARLY_STOP_IN_SWEEP,
        "CONVERGENCE_TOL": CONVERGENCE_TOL,
        "BASE_SEED": BASE_SEED,
        "GP_ALPHA_FLOOR": GP_ALPHA_FLOOR,
        "EI_XI": EI_XI,
        "RBS_FULL": RBS_FULL,
        "gp_scaled_X_passed_to_julia": False,
    }

    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in settings.items()]
    )


def print_noise_sweep_table(all_results):
    summary = build_summary_dataframe(all_results)

    print("\n===== NOISE SWEEP SUMMARY =====\n")
    print("noise      n_exp    methanol_best    rel_param_error_norm")
    print("----------------------------------------------------------")

    for _, row in summary.iterrows():
        print(
            f"{row['noise']:>5s}   "
            f"{int(row['n_experiments']):5d}   "
            f"{row['best_ch3oh_outlet_fraction']:13.6f}   "
            f"{row['rms_relative_parameter_error']:20.3e}"
        )


def export_results_to_excel(all_results, results_dir):
    excel_path = results_dir / "variable_explorer_data.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        build_settings_dataframe(results_dir).to_excel(
            writer,
            sheet_name="settings",
            index=False,
        )
        build_summary_dataframe(all_results).to_excel(
            writer,
            sheet_name="noise_summary",
            index=False,
        )
        build_final_parameters_dataframe(all_results).to_excel(
            writer,
            sheet_name="final_parameters",
            index=False,
        )
        build_parameter_history_dataframe(all_results).to_excel(
            writer,
            sheet_name="parameter_history",
            index=False,
        )
        build_design_dataframe(all_results).to_excel(
            writer,
            sheet_name="selected_designs",
            index=False,
        )
        build_y_output_dataframe(all_results).to_excel(
            writer,
            sheet_name="Y_outputs_flat",
            index=False,
        )

    return excel_path


def write_summary_text(all_results, results_dir):
    summary = build_summary_dataframe(all_results)
    lines = [
        "METHANOL BOED NOISE SWEEP SUMMARY",
        "",
        f"Results folder: {results_dir}",
        f"Noise levels: {', '.join(noise_label(noise) for noise in NOISE_LEVELS)}",
        f"N_INIT: {N_INIT}",
        f"MAX_EXPERIMENTS: {MAX_EXPERIMENTS}",
        f"N_CANDIDATES: {N_CANDIDATES}",
        f"N_REPEATS: {N_REPEATS}",
        f"RBS_FULL: {RBS_FULL}",
        "",
        "Estimator input check:",
        "  experiments() receives each selected design as Y_in=(6, 1), "
        "Temp=(1,), P_total=(1,).",
        "  parameter_estimator() receives accumulated physical design arrays: "
        "Y_in=(6, Nexps), Temp=(Nexps,), P_total=(Nexps,), "
        "Y_out=(N_repeats, 6, Nexps).",
        "  X_scaled is used only by the Python GP/EI surrogate and is not sent "
        "to Julia.",
        "",
        "noise      n_exp    methanol_best    rms_relative_parameter_error",
        "----------------------------------------------------------------",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['noise']:>5s}   "
            f"{int(row['n_experiments']):5d}   "
            f"{row['best_ch3oh_outlet_fraction']:13.6f}   "
            f"{row['rms_relative_parameter_error']:28.3e}"
        )

    summary_path = results_dir / "noise_sweep_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def plot_methanol_vs_experiments(all_results, results_dir=None):
    fig = plt.figure(figsize=(10, 6))

    for result in all_results:
        y = np.asarray(result["y"], dtype=float)
        exp_numbers = np.arange(1, len(y) + 1)
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])

        plt.plot(
            exp_numbers,
            y,
            linewidth=1.8,
            markersize=marker_size_for_noise(result["noise"]),
            markerfacecolor="none",
            markeredgewidth=1.8,
            label=label,
            **style,
        )

    plt.xlabel("Experiment number")
    plt.ylabel("CH3OH outlet mass fraction")
    plt.title("CH3OH Outlet Fraction Across Selected Experiments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "01_ch3oh_outlet_vs_experiment.png")


def plot_methanol_vs_design(all_results, results_dir=None):
    fig = plt.figure(figsize=(10, 6))
    for result in all_results:
        X = result["X"]
        y = result["y"]
        style = style_for_noise(result["noise"])
        x_display = X[:, 0] + display_offset_for_noise(result["noise"], width=4e-4)
        plt.scatter(
            x_display,
            y,
            s=scatter_size_for_noise(result["noise"]),
            alpha=0.9,
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            linewidths=1.8,
            label=f"noise {noise_label(result['noise'])}",
        )
    plt.xlabel("CO2 inlet mass fraction")
    plt.ylabel("CH3OH outlet mass fraction")
    plt.title("CH3OH Outlet Fraction vs CO2 Inlet")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "02_ch3oh_outlet_vs_co2.png")

    fig = plt.figure(figsize=(10, 6))
    for result in all_results:
        X = result["X"]
        y = result["y"]
        style = style_for_noise(result["noise"])
        x_display = X[:, -2] + display_offset_for_noise(result["noise"], width=0.12)
        plt.scatter(
            x_display,
            y,
            s=scatter_size_for_noise(result["noise"]),
            alpha=0.9,
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            linewidths=1.8,
            label=f"noise {noise_label(result['noise'])}",
        )
    plt.xlabel("Temperature (K)")
    plt.ylabel("CH3OH outlet mass fraction")
    plt.title("CH3OH Outlet Fraction vs Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "03_ch3oh_outlet_vs_temperature.png")

    fig = plt.figure(figsize=(10, 6))
    for result in all_results:
        X = result["X"]
        y = result["y"]
        style = style_for_noise(result["noise"])
        x_display = X[:, -1] + display_offset_for_noise(result["noise"], width=0.04)
        plt.scatter(
            x_display,
            y,
            s=scatter_size_for_noise(result["noise"]),
            alpha=0.9,
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            linewidths=1.8,
            label=f"noise {noise_label(result['noise'])}",
        )
    plt.xlabel("Total pressure (bar)")
    plt.ylabel("CH3OH outlet mass fraction")
    plt.title("CH3OH Outlet Fraction vs Total Pressure")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "04_ch3oh_outlet_vs_pressure.png")


def plot_3d_design_space(all_results, results_dir=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for result in all_results:
        X = result["X"]
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])
        ax.scatter(
            X[:, 0],
            X[:, -2],
            X[:, -1],
            s=scatter_size_for_noise(result["noise"]),
            alpha=0.9,
            marker=style["marker"],
            color=style["color"],
            label=label,
        )

    ax.set_xlabel("CO2 inlet mass fraction")
    ax.set_ylabel("Temperature (K)")
    ax.set_zlabel("Total pressure (bar)")
    ax.set_title("Selected Design Space: CO2, Temperature, Pressure")
    ax.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "05_selected_design_space_3d.png")


def plot_parameters_mixed_log(all_results, results_dir=None):
    for param_index in range(N_UNKNOWN_PARAMETERS):
        fig = plt.figure(figsize=(8, 5))
        true_value, ylabel, transform_name = transformed_parameter_values(
            [TRUE_PARAMS[param_index]],
            param_index,
        )
        initial_value, _, _ = transformed_parameter_values(
            [INITIAL_GUESS_PHYSICAL[param_index]],
            param_index,
        )

        for result in all_results:
            params = np.asarray(result["params"], dtype=float)
            exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
            label = f"noise {noise_label(result['noise'])}"
            style = style_for_noise(result["noise"])
            values, _, _ = transformed_parameter_values(
                params[:, param_index],
                param_index,
            )

            plt.plot(
                exp_counts,
                values,
                linewidth=1.8,
                markersize=marker_size_for_noise(result["noise"]),
                markerfacecolor="none",
                markeredgewidth=1.8,
                label=label,
                **style,
            )

        plt.axhline(
            true_value[0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="true value",
        )
        plt.axhline(
            initial_value[0],
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="initial guess",
        )
        plt.title(
            f"p{param_index + 1}: {PARAMETER_SHORT_NAMES[param_index]} "
            f"{transform_name} Convergence"
        )
        plt.xlabel("Total experiments used")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"06_p{param_index + 1:02d}_mixed_log_convergence.png"
        save_and_show(fig, results_dir, filename)


def plot_parameter_errors(all_results, results_dir=None):
    for param_index in range(N_UNKNOWN_PARAMETERS):
        fig = plt.figure(figsize=(8, 5))

        for result in all_results:
            params = np.asarray(result["params"], dtype=float)
            exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
            label = f"noise {noise_label(result['noise'])}"
            style = style_for_noise(result["noise"])
            true_val = TRUE_PARAMS[param_index]
            rel_err = (params[:, param_index] - true_val) / max(abs(true_val), 1e-12)

            plt.plot(
                exp_counts,
                rel_err,
                linewidth=1.8,
                markersize=marker_size_for_noise(result["noise"]),
                markerfacecolor="none",
                markeredgewidth=1.8,
                label=label,
                **style,
            )

        plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        plt.yscale("symlog", linthresh=1e-3)
        plt.title(
            f"p{param_index + 1}: {PARAMETER_SHORT_NAMES[param_index]} "
            "Relative Error"
        )
        plt.xlabel("Total experiments used")
        plt.ylabel("(estimate - true) / |true|")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"07_p{param_index + 1:02d}_relative_error.png"
        save_and_show(fig, results_dir, filename)


def plot_mixed_log_error(all_results, results_dir=None):
    fig = plt.figure(figsize=(8, 5))
    true_mixed = mixed_log_matrix(TRUE_PARAMS.reshape(1, -1))[0]

    for result in all_results:
        params = np.asarray(result["params"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])
        params_mixed = mixed_log_matrix(params)
        errors = np.linalg.norm(params_mixed - true_mixed, axis=1) / np.sqrt(
            N_UNKNOWN_PARAMETERS
        )

        plt.plot(
            exp_counts,
            errors,
            linewidth=1.8,
            markersize=marker_size_for_noise(result["noise"]),
            markerfacecolor="none",
            markeredgewidth=1.8,
            label=label,
            **style,
        )

    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("RMS mixed-log parameter error")
    plt.title("Overall Parameter Error in Mixed Log Space")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "08_overall_mixed_log_parameter_error.png")


def export_all_outputs(all_results, results_dir, shared_initial_design):
    excel_path = export_results_to_excel(all_results, results_dir)
    summary_path = write_summary_text(all_results, results_dir)

    print("\nSaved run outputs:")
    print(f"  Results folder: {results_dir}")
    print(f"  Excel data: {excel_path}")
    print(f"  Summary: {summary_path}")


def plot_noise_sweep(all_results, results_dir=None):
    print_overlap_diagnostics(all_results)
    plot_methanol_vs_experiments(all_results, results_dir)
    plot_methanol_vs_design(all_results, results_dir)
    plot_3d_design_space(all_results, results_dir)
    plot_parameters_mixed_log(all_results, results_dir)
    plot_parameter_errors(all_results, results_dir)
    plot_mixed_log_error(all_results, results_dir)



# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":
    all_results = []
    results_dir = RESULTS_DIR

    print(f"\nSaving all outputs to: {results_dir}\n")

    initial_rng = np.random.default_rng(BASE_SEED)
    shared_initial_design = generate_initial_design(N_INIT, rng=initial_rng)

    for noise in NOISE_LEVELS:
        print("\n")
        print("=" * 60)
        print(f"RUNNING METHANOL BOED FOR NOISE = {noise:.0e}")
        print("=" * 60)

        X, y, Y_full, param_history, param_exp_counts = bayesian_optimization(
            noise_level=noise,
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
    export_all_outputs(all_results, results_dir, shared_initial_design)
    plot_noise_sweep(all_results, results_dir)


    