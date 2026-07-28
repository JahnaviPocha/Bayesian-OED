# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:12:18 2026

@author: jahna
"""

"""
Corrected Fisher-information Bayesian OED driver for the 2-parameter ROM case.

Main corrections and improvements:
- fixes the noise-study bug: noise changes STD_DATA, not N_REPEATS
- keeps observed experiment data fixed once collected
- uses actual experiment counts on convergence plots
- uses the same initial design for all noise levels
- replaces the old covariance-of-repeated-estimates proxy with a local
  D-optimal Fisher information objective based on finite-difference
  sensitivities dY_out / dk
- tracks cumulative information gain of the selected design
- uses combined plots for all noise levels

Expected Julia tensor shape:
    Y_out = (N_repeats, Nspec, Nexps)

Put this file in inverse_prob_julia/ next to call_to_KPE_code.py before running.
"""
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from call_to_KPE_code import experiments, parameter_estimator


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3
N_REACTIONS = 1
N_UNKNOWN_PARAMETERS = 2
TARGET_SPECIES_INDEX = 2

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

TRUE_K = np.array([4000.0, 4000.0], dtype=float)
INITIAL_GUESS = np.array([1000.0, 1000.0], dtype=float)

PARAMETER_NAMES = ["k1", "k2"]
SPECIES_NAMES = ["species 1", "species 2", "species 3"]

P_TOTAL = 50.0
RATIO = 0.1
SCALE = 1.0
STOICHIOMETRY = np.array([[-2.0, -1.0, 2.0]], dtype=float)

N_REPEATS = 10
NOISE_LEVELS = [1e-3, 1e-5, 1e-7]

N_INIT = 10
MAX_EXPERIMENTS = 100
N_CANDIDATES = 200
MAX_BATCH_SIZE = 2

ALLOW_EARLY_STOP_IN_SWEEP = False
CONVERGENCE_TOL = 1e-3
BASE_SEED = 12345

# In synthetic validation it is fine to use the true value as the nominal
# sensitivity point. For real experiments, use your best prior estimate.
NOMINAL_K_FOR_INFORMATION = TRUE_K.copy()

FD_REL_STEP = 1e-3
FD_ABS_STEP = 1e-3
FIM_NOISE_FLOOR = 1e-12

PRIOR_STD = np.array([1.0e4, 1.0e4], dtype=float)
PRIOR_PRECISION = np.diag(1.0 / (PRIOR_STD ** 2))

GP_ALPHA = 1e-8
EI_XI = 0.01
MIN_SCALED_DISTANCE = 0.03

RBS_FULL = False

RESULTS_FOLDER_NAME = "bayes_design_ROM_FIO_asymptotic"
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
    """Convert independent inlet mass fractions to the full species vector."""
    Y_partial = np.asarray(Y_partial, dtype=float)
    last = 1.0 - np.sum(Y_partial)

    if last < 0.0:
        return None

    return np.append(Y_partial, last)


def decode_design_vector(x):
    """Decode x = [Y1, Y2, Temp]."""
    x = np.asarray(x, dtype=float)

    if x.size != NSPEC:
        return None, None

    Y_partial = x[:-1]
    temp = float(x[-1])
    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, temp


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


def validate_deterministic_shape(Yexp):
    """Check deterministic single-repeat Julia tensor shape."""
    arr = np.asarray(Yexp, dtype=float)
    expected = (1, NSPEC, 1)

    if arr.shape != expected:
        raise ValueError(
            f"deterministic Yexp has shape {arr.shape}, expected {expected}."
        )

    return arr


# ============================================================
# JULIA MODEL WRAPPERS
# ============================================================

_MODEL_OUTPUT_CACHE = {}
_SENSITIVITY_CACHE = {}


def _cache_key_vector(v, decimals=10):
    return tuple(np.round(np.asarray(v, dtype=float), decimals=decimals))


def deterministic_output(x, k):
    """
    Deterministic outlet vector Y_out(x, k), no added noise.

    Uses Julia experiments() with std_data=0 and N_repeats=1.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(k, dtype=float)
    key = (_cache_key_vector(x, 12), _cache_key_vector(k, 8))

    if key in _MODEL_OUTPUT_CACHE:
        return _MODEL_OUTPUT_CACHE[key].copy()

    Y_in, temp = decode_design_vector(x)

    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Yexp = experiments(
        scale=SCALE,
        Y_in=np.asarray(Y_in, dtype=float).reshape(NSPEC, 1),
        Temp=np.asarray([temp], dtype=float),
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=1,
        std_data=0.0,
        Nspec=NSPEC,
        k_true=k.tolist(),
    )

    Yexp = validate_deterministic_shape(Yexp)
    y_out = np.asarray(Yexp[0, :, 0], dtype=float)

    _MODEL_OUTPUT_CACHE[key] = y_out.copy()
    return y_out


def finite_difference_sensitivities(x, k_ref):
    """Compute J = dY_out / dk at one design."""
    x = np.asarray(x, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    key = (
        _cache_key_vector(x, 12),
        _cache_key_vector(k_ref, 8),
        FD_REL_STEP,
        FD_ABS_STEP,
    )

    if key in _SENSITIVITY_CACHE:
        return _SENSITIVITY_CACHE[key].copy()

    J = np.zeros((NSPEC, N_UNKNOWN_PARAMETERS), dtype=float)
    y0 = None

    for j in range(N_UNKNOWN_PARAMETERS):
        step = max(abs(k_ref[j]) * FD_REL_STEP, FD_ABS_STEP)
        k_plus = k_ref.copy()
        k_minus = k_ref.copy()
        k_plus[j] += step
        k_minus[j] -= step

        if k_minus[j] > 0.0:
            y_plus = deterministic_output(x, k_plus)
            y_minus = deterministic_output(x, k_minus)
            J[:, j] = (y_plus - y_minus) / (2.0 * step)
        else:
            if y0 is None:
                y0 = deterministic_output(x, k_ref)
            y_plus = deterministic_output(x, k_plus)
            J[:, j] = (y_plus - y0) / step

    _SENSITIVITY_CACHE[key] = J.copy()
    return J


def fisher_information_matrix(x, k_ref, noise_level):
    J = finite_difference_sensitivities(x, k_ref)
    sigma = max(float(noise_level), FIM_NOISE_FLOOR)
    weight = N_REPEATS / (sigma ** 2)
    return weight * (J.T @ J)


def information_gain_from_F(F):
    """D-optimal information gain relative to the weak prior precision."""
    sign0, logdet0 = np.linalg.slogdet(PRIOR_PRECISION)
    sign1, logdet1 = np.linalg.slogdet(PRIOR_PRECISION + F)

    if sign0 <= 0 or sign1 <= 0:
        return -1.0e12

    return 0.5 * (logdet1 - logdet0)


def point_information_objective(x, noise_level, k_ref=None):
    """Pointwise D-optimal information score for one experiment."""
    if k_ref is None:
        k_ref = NOMINAL_K_FOR_INFORMATION

    Y_in, _ = decode_design_vector(x)
    if Y_in is None:
        return -1.0e12, np.zeros((N_UNKNOWN_PARAMETERS, N_UNKNOWN_PARAMETERS))

    try:
        F = fisher_information_matrix(x, k_ref, noise_level)
        score = information_gain_from_F(F)
    except Exception as exc:
        print(f"Information objective failed at x={x}: {exc}")
        return -1.0e12, np.zeros((N_UNKNOWN_PARAMETERS, N_UNKNOWN_PARAMETERS))

    return float(score), F


def cumulative_information_gain(F_matrices):
    if not F_matrices:
        return 0.0

    F_total = np.sum(np.asarray(F_matrices, dtype=float), axis=0)
    return information_gain_from_F(F_total)


def run_noisy_experiment(x, noise_level):
    """Run one noisy experiment for parameter estimation."""
    Y_in, temp = decode_design_vector(x)

    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Yexp = experiments(
        scale=SCALE,
        Y_in=np.asarray(Y_in, dtype=float).reshape(NSPEC, 1),
        Temp=np.asarray([temp], dtype=float),
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
        k_true=TRUE_K.tolist(),
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)
    y_mean = np.mean(Yexp, axis=0)[:, 0]
    target = float(y_mean[TARGET_SPECIES_INDEX])

    return target, Yexp


def estimate_parameters_from_observations(
    X,
    Y_full,
    noise_level,
    initial_guess=None,
):
    """Estimate k from already-collected observations. Does not resimulate old data."""
    X = np.asarray(X, dtype=float)
    nexps = X.shape[0]

    Y_in_all = []
    temp_all = []

    for x in X:
        Y_in, temp = decode_design_vector(x)

        if Y_in is None:
            raise ValueError(f"Invalid design vector in X: {x}")

        Y_in_all.append(Y_in)
        temp_all.append(temp)

    Y_in_all = np.asarray(Y_in_all, dtype=float).T
    temp_all = np.asarray(temp_all, dtype=float)
    Y_full = validate_yexp_shape(Y_full, nexps_expected=nexps, name="Y_full")

    if initial_guess is None:
        initial_guess = INITIAL_GUESS

    if PRINT_ESTIMATOR_SHAPES:
        print(
            "\nEstimator receives real design arrays: "
            f"Y_in={Y_in_all.shape}, "
            f"Temp={temp_all.shape}, "
            f"Y_out={Y_full.shape}"
        )

    kwargs = {
        "scale": SCALE,
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": temp_all,
        "P_total": P_TOTAL,
        "St": STOICHIOMETRY,
        "nref": 2500,
        "nreac": N_REACTIONS,
        "Nexps": nexps,
        "Y_out": Y_full,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": np.asarray(initial_guess, dtype=float),
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
    }

    k = parameter_estimator(**kwargs)
    k = np.asarray(k, dtype=float).reshape(-1)

    if k.size != N_UNKNOWN_PARAMETERS:
        raise ValueError(
            f"parameter_estimator returned {k.size} values, "
            f"expected {N_UNKNOWN_PARAMETERS}."
        )

    return k


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design(N_init=N_INIT, rng=None):
    rng = np.random.default_rng(BASE_SEED) if rng is None else rng
    X_init = []

    while len(X_init) < N_init:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [temp], dtype=float)
        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            X_init.append(x)

    return np.asarray(X_init, dtype=float)


def generate_candidates(n_candidates=N_CANDIDATES, rng=None):
    rng = np.random.default_rng(BASE_SEED) if rng is None else rng
    candidates = []

    while len(candidates) < n_candidates:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [temp], dtype=float)
        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            candidates.append(x)

    return np.asarray(candidates, dtype=float)


# ============================================================
# GP SURROGATE + EXPECTED IMPROVEMENT
# ============================================================

def scale_X(X):
    """
    Scale design variables to [0, 1] for GP stability.

    These scaled values are used only by the Python GP/EI surrogate. They are
    never passed to Julia experiments() or parameter_estimator().
    """
    X = np.asarray(X, dtype=float)
    X_scaled = np.zeros_like(X)

    for i in range(NSPEC - 1):
        lb, ub = Y_BOUNDS[i]
        X_scaled[:, i] = (X[:, i] - lb) / (ub - lb)

    lb, ub = TEMP_BOUNDS
    X_scaled[:, -1] = (X[:, -1] - lb) / (ub - lb)

    return X_scaled


def train_gp(X, y):
    X_scaled = scale_X(X)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=BASE_SEED,
    )

    gp.fit(X_scaled, y)
    return gp


def expected_improvement(X_candidates, gp, y_best, xi=EI_XI):
    X_scaled = scale_X(X_candidates)
    mu, sigma = gp.predict(X_scaled, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    sigma_safe = np.maximum(sigma, 1e-15)

    improvement = mu - y_best - xi
    Z = improvement / sigma_safe
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0

    return ei.ravel()


def filter_far_candidates(X_candidates, X_existing):
    """Avoid selecting nearly duplicate designs."""
    cand_scaled = scale_X(X_candidates)
    existing_scaled = scale_X(X_existing)

    dists = np.linalg.norm(
        cand_scaled[:, None, :] - existing_scaled[None, :, :],
        axis=2,
    )
    min_dists = np.min(dists, axis=1)
    mask = min_dists >= MIN_SCALED_DISTANCE

    if not np.any(mask):
        return X_candidates

    return X_candidates[mask]


# ============================================================
# BO LOOP
# ============================================================

def relative_parameter_errors(params):
    params = np.asarray(params, dtype=float)
    denom = np.maximum(np.abs(TRUE_K), 1e-30)
    return (params - TRUE_K) / denom


def check_parameter_convergence(param_history, tol=CONVERGENCE_TOL):
    if len(param_history) < 2:
        return False

    prev = np.asarray(param_history[-2], dtype=float)
    curr = np.asarray(param_history[-1], dtype=float)
    denom = np.maximum(np.abs(prev), np.abs(TRUE_K))
    denom = np.maximum(denom, 1e-30)
    delta = np.linalg.norm((curr - prev) / denom) / np.sqrt(curr.size)

    print(f"Relative parameter change: {delta:.4e}")
    return delta < tol


def BO(
    noise_level,
    N_init=N_INIT,
    max_experiments=MAX_EXPERIMENTS,
    n_candidates=N_CANDIDATES,
    allow_early_stop=ALLOW_EARLY_STOP_IN_SWEEP,
    tol=CONVERGENCE_TOL,
    rng_seed=BASE_SEED,
    initial_design=None,
):
    print("\n=== INITIAL ROM/RBS FIO EXPERIMENTS ===")
    print(f"noise_level = {noise_level:.0e}")

    rng = np.random.default_rng(rng_seed)

    if initial_design is None:
        X = generate_initial_design(N_init, rng=rng)
    else:
        X = np.asarray(initial_design, dtype=float).copy()

    y_info = []
    y_target = []
    Y_tensor_list = []
    F_matrices = []

    for i, x in enumerate(X):
        info, F = point_information_objective(
            x,
            noise_level=noise_level,
            k_ref=NOMINAL_K_FOR_INFORMATION,
        )
        target, Yexp = run_noisy_experiment(x, noise_level=noise_level)

        y_info.append(info)
        y_target.append(target)
        F_matrices.append(F)
        Y_tensor_list.append(Yexp)

        print(f"Init Exp {i + 1}: x={x}")
        print(f"  point information = {info:.6f}")
        print(f"  target output = {target:.6f}")

    y_info = np.asarray(y_info, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    n_initial = len(X)
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=n_initial,
        name="Y_full",
    )

    k = estimate_parameters_from_observations(
        X,
        Y_full,
        noise_level=noise_level,
        initial_guess=INITIAL_GUESS,
    )
    param_history = [k]
    param_exp_counts = [len(X)]
    total_info_history = [cumulative_information_gain(F_matrices)]

    print("Initial k =", k)
    print(f"Initial cumulative information = {total_info_history[-1]:.6f}")

    print("\n=== ROM/RBS FIO BO START ===")

    while len(X) < max_experiments:
        gp = train_gp(X, y_info)
        X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
        X_candidates = filter_far_candidates(X_candidates, X)

        if len(X_candidates) == 0:
            print(
                "Warning: distance filter removed all candidates. "
                "Regenerating candidates without the distance filter."
            )
            X_candidates = generate_candidates(
                n_candidates=max(n_candidates, MAX_BATCH_SIZE),
                rng=rng,
            )

        ei = expected_improvement(X_candidates, gp, np.max(y_info))
        sorted_idx = np.argsort(ei)[::-1]
        remaining = max_experiments - len(X)
        top_k = min(MAX_BATCH_SIZE, remaining, len(sorted_idx))
        X_batch = X_candidates[sorted_idx[:top_k]]

        if len(X_batch) == 0:
            raise RuntimeError(
                "No valid BO candidates were available. Increase N_CANDIDATES "
                "or reduce MIN_SCALED_DISTANCE."
            )

        print(f"\n--- BO Iteration: evaluating top-{len(X_batch)} candidates ---")

        y_new_list = []
        Yexp_new_list = []

        for x_next in X_batch:
            y_next, Yexp_next = run_noisy_experiment(
                x_next,
                noise_level=noise_level,
            )
            y_new_list.append(y_next)
            Yexp_new_list.append(Yexp_next)
            print(f"  Candidate done: target={y_next:.6f}")

        info_new_list = []
        F_new_list = []

        for x_next in X_batch:
            info_next, F_next = point_information_objective(
                x_next,
                noise_level=noise_level,
                k_ref=NOMINAL_K_FOR_INFORMATION,
            )
            info_new_list.append(info_next)
            F_new_list.append(F_next)

        X = np.vstack([X, X_batch])
        y_info = np.append(y_info, np.asarray(info_new_list, dtype=float))
        y_target = np.append(y_target, np.asarray(y_new_list, dtype=float))
        F_matrices.extend(F_new_list)
        Y_full = validate_yexp_shape(
            np.concatenate([Y_full] + Yexp_new_list, axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        k = estimate_parameters_from_observations(
            X,
            Y_full,
            noise_level=noise_level,
            initial_guess=param_history[-1],
        )
        param_history.append(k)
        param_exp_counts.append(len(X))
        total_info_history.append(cumulative_information_gain(F_matrices))

        print("\n--------------------------------")
        print(f"Batch update: total experiments = {len(X)}")
        print("--------------------------------")
        print(f"Evaluated {len(X_batch)} candidates this iteration")
        print(f"Cumulative information = {total_info_history[-1]:.6f}")
        print("Estimated k =", k)

        if allow_early_stop and check_parameter_convergence(param_history, tol):
            print("\nConvergence reached -> stopping early.")
            break

    return {
        "X": X,
        "point_information": y_info,
        "target_output": y_target,
        "Y_full": Y_full,
        "F_matrices": np.asarray(F_matrices, dtype=float),
        "param_history": np.asarray(param_history, dtype=float),
        "param_exp_counts": np.asarray(param_exp_counts, dtype=int),
        "total_info_history": np.asarray(total_info_history, dtype=float),
    }


def run_noise_study():
    results = {}
    initial_rng = np.random.default_rng(BASE_SEED)
    shared_initial_design = generate_initial_design(N_INIT, rng=initial_rng)

    for noise in NOISE_LEVELS:
        print("\n")
        print("=" * 60)
        print(f"RUNNING ROM/RBS FIO CASE sigma = {noise:.0e}")
        print("=" * 60)

        result = BO(
            noise_level=noise,
            N_init=N_INIT,
            max_experiments=MAX_EXPERIMENTS,
            n_candidates=N_CANDIDATES,
            allow_early_stop=ALLOW_EARLY_STOP_IN_SWEEP,
            tol=CONVERGENCE_TOL,
            rng_seed=BASE_SEED,
            initial_design=shared_initial_design,
        )

        final_k = result["param_history"][-1]
        rel_errors = relative_parameter_errors(final_k)
        rel_norm = np.linalg.norm(rel_errors) / np.sqrt(N_UNKNOWN_PARAMETERS)
        best_idx = int(np.argmax(result["point_information"]))

        print("\n===== FINAL RESULTS SUMMARY =====")
        print(f"noise sigma = {noise:.0e}")
        print("Total experiments used:", len(result["X"]))
        print(f"RMS relative parameter error = {rel_norm:.6e}")
        print(f"k1 = {final_k[0]:.6f}")
        print(f"k2 = {final_k[1]:.6f}")
        print("Best pointwise-information experiment:")
        print(result["X"][best_idx])
        print(f"Point information = {result['point_information'][best_idx]:.6f}")
        print(f"Final cumulative information = {result['total_info_history'][-1]:.6f}")

        results[noise] = result

    return results


# ============================================================
# EXPORTS
# ============================================================

def build_settings_dataframe():
    settings = {
        "results_dir": str(RESULTS_DIR),
        "NSPEC": NSPEC,
        "N_REACTIONS": N_REACTIONS,
        "N_UNKNOWN_PARAMETERS": N_UNKNOWN_PARAMETERS,
        "TARGET_SPECIES_INDEX_python": TARGET_SPECIES_INDEX,
        "TRUE_K": ", ".join(f"{value:g}" for value in TRUE_K),
        "INITIAL_GUESS": ", ".join(f"{value:g}" for value in INITIAL_GUESS),
        "P_TOTAL": P_TOTAL,
        "RATIO": RATIO,
        "SCALE": SCALE,
        "N_REPEATS": N_REPEATS,
        "NOISE_LEVELS": ", ".join(noise_label(noise) for noise in NOISE_LEVELS),
        "N_INIT": N_INIT,
        "MAX_EXPERIMENTS": MAX_EXPERIMENTS,
        "N_CANDIDATES": N_CANDIDATES,
        "MAX_BATCH_SIZE": MAX_BATCH_SIZE,
        "ALLOW_EARLY_STOP_IN_SWEEP": ALLOW_EARLY_STOP_IN_SWEEP,
        "CONVERGENCE_TOL": CONVERGENCE_TOL,
        "BASE_SEED": BASE_SEED,
        "FD_REL_STEP": FD_REL_STEP,
        "FD_ABS_STEP": FD_ABS_STEP,
        "FIM_NOISE_FLOOR": FIM_NOISE_FLOOR,
        "PRIOR_STD": ", ".join(f"{value:g}" for value in PRIOR_STD),
        "GP_ALPHA": GP_ALPHA,
        "EI_XI": EI_XI,
        "MIN_SCALED_DISTANCE": MIN_SCALED_DISTANCE,
        "RBS_FULL": RBS_FULL,
        "gp_scaled_X_passed_to_julia": False,
    }
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in settings.items()]
    )


def build_summary_dataframe(results):
    rows = []

    for noise, data in results.items():
        final_k = np.asarray(data["param_history"][-1], dtype=float)
        rel_errors = relative_parameter_errors(final_k)
        best_info_idx = int(np.argmax(data["point_information"]))
        best_target_idx = int(np.argmax(data["target_output"]))

        rows.append(
            {
                "noise": noise_label(noise),
                "sigma": noise,
                "n_experiments": len(data["X"]),
                "best_point_information": data["point_information"][best_info_idx],
                "best_point_information_experiment": best_info_idx + 1,
                "best_target_output": data["target_output"][best_target_idx],
                "best_target_output_experiment": best_target_idx + 1,
                "final_cumulative_information": data["total_info_history"][-1],
                "k1_final": final_k[0],
                "k2_final": final_k[1],
                "rms_relative_parameter_error": np.linalg.norm(rel_errors)
                / np.sqrt(N_UNKNOWN_PARAMETERS),
            }
        )

    return pd.DataFrame(rows)


def build_final_parameters_dataframe(results):
    rows = []

    for noise, data in results.items():
        final_k = np.asarray(data["param_history"][-1], dtype=float)
        rel_errors = relative_parameter_errors(final_k)
        abs_errors = final_k - TRUE_K

        for i, name in enumerate(PARAMETER_NAMES):
            rows.append(
                {
                    "noise": noise_label(noise),
                    "parameter": name,
                    "true_value": TRUE_K[i],
                    "initial_guess": INITIAL_GUESS[i],
                    "final_estimate": final_k[i],
                    "absolute_error": abs_errors[i],
                    "relative_error": rel_errors[i],
                }
            )

    return pd.DataFrame(rows)


def build_parameter_history_dataframe(results):
    rows = []

    for noise, data in results.items():
        params = np.asarray(data["param_history"], dtype=float)
        exp_counts = np.asarray(data["param_exp_counts"], dtype=int)

        for row_index, exp_count in enumerate(exp_counts):
            rel_errors = relative_parameter_errors(params[row_index])
            abs_errors = params[row_index] - TRUE_K

            for i, name in enumerate(PARAMETER_NAMES):
                rows.append(
                    {
                        "noise": noise_label(noise),
                        "total_experiments_used": int(exp_count),
                        "parameter": name,
                        "estimate": params[row_index, i],
                        "true_value": TRUE_K[i],
                        "absolute_error": abs_errors[i],
                        "relative_error": rel_errors[i],
                    }
                )

    return pd.DataFrame(rows)


def build_design_dataframe(results):
    rows = []

    for noise, data in results.items():
        for exp_index, x in enumerate(np.asarray(data["X"], dtype=float)):
            Y_full, temp = decode_design_vector(x)
            row = {
                "noise": noise_label(noise),
                "experiment_number": exp_index + 1,
                "point_information": data["point_information"][exp_index],
                "target_output_species_3": data["target_output"][exp_index],
                "temperature_K": temp,
            }

            for species_index, species_name in enumerate(SPECIES_NAMES):
                row[f"{species_name}_inlet_mass_fraction"] = Y_full[species_index]

            rows.append(row)

    return pd.DataFrame(rows)


def build_information_history_dataframe(results):
    rows = []

    for noise, data in results.items():
        for i, exp_count in enumerate(data["param_exp_counts"]):
            rows.append(
                {
                    "noise": noise_label(noise),
                    "total_experiments_used": int(exp_count),
                    "cumulative_information": data["total_info_history"][i],
                }
            )

    return pd.DataFrame(rows)


def build_y_outputs_dataframe(results):
    rows = []

    for noise, data in results.items():
        Y_full = np.asarray(data["Y_full"], dtype=float)

        for repeat_index in range(Y_full.shape[0]):
            for species_index, species_name in enumerate(SPECIES_NAMES):
                for exp_index in range(Y_full.shape[2]):
                    rows.append(
                        {
                            "noise": noise_label(noise),
                            "repeat_number": repeat_index + 1,
                            "species_index": species_index + 1,
                            "species": species_name,
                            "experiment_number": exp_index + 1,
                            "outlet_fraction": Y_full[
                                repeat_index,
                                species_index,
                                exp_index,
                            ],
                        }
                    )

    return pd.DataFrame(rows)


def export_results_to_excel(results):
    excel_path = RESULTS_DIR / "variable_explorer_data.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        build_settings_dataframe().to_excel(writer, sheet_name="settings", index=False)
        build_summary_dataframe(results).to_excel(
            writer,
            sheet_name="noise_summary",
            index=False,
        )
        build_final_parameters_dataframe(results).to_excel(
            writer,
            sheet_name="final_parameters",
            index=False,
        )
        build_parameter_history_dataframe(results).to_excel(
            writer,
            sheet_name="parameter_history",
            index=False,
        )
        build_design_dataframe(results).to_excel(
            writer,
            sheet_name="selected_designs",
            index=False,
        )
        build_information_history_dataframe(results).to_excel(
            writer,
            sheet_name="information_history",
            index=False,
        )
        build_y_outputs_dataframe(results).to_excel(
            writer,
            sheet_name="Y_outputs_flat",
            index=False,
        )

    return excel_path


def write_summary_text(results):
    lines = [
        "===== ROM/RBS FIO NOISE SWEEP SUMMARY =====",
        "",
        f"Results folder: {RESULTS_DIR}",
        f"Noise levels: {', '.join(noise_label(noise) for noise in NOISE_LEVELS)}",
        f"N_INIT: {N_INIT}",
        f"MAX_EXPERIMENTS: {MAX_EXPERIMENTS}",
        f"N_CANDIDATES: {N_CANDIDATES}",
        f"N_REPEATS: {N_REPEATS}",
        f"RBS_FULL: {RBS_FULL}",
        "",
    ]

    for noise, data in results.items():
        final_k = np.asarray(data["param_history"][-1], dtype=float)
        rel_errors = relative_parameter_errors(final_k)
        rel_norm = np.linalg.norm(rel_errors) / np.sqrt(N_UNKNOWN_PARAMETERS)
        best_info_idx = int(np.argmax(data["point_information"]))
        best_target_idx = int(np.argmax(data["target_output"]))

        lines.extend(
            [
                "=" * 60,
                "===== FINAL RESULTS SUMMARY =====",
                "",
                f"noise sigma = {noise:.0e}",
                f"Total experiments used: {len(data['X'])}",
                f"RMS relative parameter error = {rel_norm:.6e}",
                "",
                "Final estimated parameters:",
            ]
        )

        for i, value in enumerate(final_k):
            lines.append(
                f"  {PARAMETER_NAMES[i]} = {value:.6f} "
                f"true={TRUE_K[i]:.6f} "
                f"rel_error={rel_errors[i]:+.3e}"
            )

        lines.extend(
            [
                "",
                "Best pointwise-information experiment:",
                f"  X = {np.array2string(data['X'][best_info_idx], precision=8)}",
                f"  Point information = {data['point_information'][best_info_idx]:.6f}",
                "",
                "Best target-output experiment:",
                f"  X = {np.array2string(data['X'][best_target_idx], precision=8)}",
                f"  Target output = {data['target_output'][best_target_idx]:.6f}",
                "",
                f"Final cumulative information = {data['total_info_history'][-1]:.6f}",
                "",
            ]
        )

    summary_path = RESULTS_DIR / "final_results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def export_all_outputs(results):
    excel_path = export_results_to_excel(results)
    summary_path = write_summary_text(results)

    print("\nSaved outputs:")
    print(f"  Folder: {RESULTS_DIR}")
    print(f"  Excel: {excel_path}")
    print(f"  Summary: {summary_path}")


# ============================================================
# PLOTS
# ============================================================

def _apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def style_for_noise(noise):
    colors = {1e-3: "tab:blue", 1e-5: "tab:orange", 1e-7: "tab:green"}
    markers = {1e-3: "o", 1e-5: "s", 1e-7: "^"}
    linestyles = {1e-3: "-", 1e-5: "--", 1e-7: "-."}

    return {
        "color": colors.get(noise),
        "marker": markers.get(noise, "o"),
        "linestyle": linestyles.get(noise, "-"),
    }


def plot_k_convergence_all(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        label = f"sigma={noise_label(noise)}"
        style = style_for_noise(noise)

        axes[0].plot(exp_counts, params[:, 0], linewidth=2, label=label, **style)
        axes[1].plot(exp_counts, params[:, 1], linewidth=2, label=label, **style)

    for i, ax in enumerate(axes):
        ax.axhline(TRUE_K[i], color="black", linestyle="--", linewidth=2)
        ax.axhline(INITIAL_GUESS[i], color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"Estimated {PARAMETER_NAMES[i]}")
        ax.set_title(f"{PARAMETER_NAMES[i]} Convergence")
        ax.grid(True)
        _apply_plain_axis(ax)

    axes[0].legend()
    fig.tight_layout()
    save_and_show(fig, RESULTS_DIR, "01_parameter_convergence.png")


def plot_parameter_error(results):
    fig = plt.figure(figsize=(9, 5))

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        rel_norm = np.linalg.norm(relative_parameter_errors(params), axis=1) / np.sqrt(
            N_UNKNOWN_PARAMETERS
        )
        plt.plot(
            exp_counts,
            rel_norm,
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style_for_noise(noise),
        )

    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("RMS relative parameter error")
    plt.title("ROM/RBS FIO Relative Parameter Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, RESULTS_DIR, "02_rms_relative_parameter_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        rel_err = relative_parameter_errors(params)
        style = style_for_noise(noise)

        axes[0].plot(
            exp_counts,
            rel_err[:, 0],
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style,
        )
        axes[1].plot(
            exp_counts,
            rel_err[:, 1],
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style,
        )

    for i, ax in enumerate(axes):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"Relative error in {PARAMETER_NAMES[i]}")
        ax.set_title(f"Relative Error: {PARAMETER_NAMES[i]}")
        ax.grid(True)
        ax.set_yscale("symlog", linthresh=1e-6)

    axes[0].legend()
    fig.tight_layout()
    save_and_show(fig, RESULTS_DIR, "03_parameter_relative_errors.png")


def plot_information_gain(results):
    fig = plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        plt.plot(
            data["param_exp_counts"],
            data["total_info_history"],
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style_for_noise(noise),
        )

    plt.xlabel("Total experiments used")
    plt.ylabel("Cumulative D-optimal information gain")
    plt.title("ROM/RBS FIO Information Gain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, RESULTS_DIR, "04_cumulative_information_gain.png")


def plot_point_information(results):
    fig = plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        x_axis = np.arange(1, len(data["point_information"]) + 1)
        best_info = np.maximum.accumulate(data["point_information"])

        plt.plot(
            x_axis,
            best_info,
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style_for_noise(noise),
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Best pointwise information")
    plt.title("Best ROM/RBS FIO Pointwise Information Found by BO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, RESULTS_DIR, "05_best_point_information.png")


def plot_target_outputs(results):
    fig = plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        x_axis = np.arange(1, len(data["target_output"]) + 1)

        plt.plot(
            x_axis,
            data["target_output"],
            linewidth=2,
            label=f"sigma={noise_label(noise)}",
            **style_for_noise(noise),
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Target species outlet fraction")
    plt.title("Target Output at Selected ROM/RBS FIO Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, RESULTS_DIR, "06_target_outputs.png")


def plot_designs(results):
    fig = plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        X = data["X"]
        plt.plot(
            X[:, 0],
            X[:, -1],
            linewidth=1.5,
            label=f"sigma={noise_label(noise)}",
            **style_for_noise(noise),
        )

    plt.xlabel("Y1 inlet mass fraction")
    plt.ylabel("Temperature (K)")
    plt.title("Selected Experimental Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, RESULTS_DIR, "07_selected_designs.png")


def plot_all(results):
    plot_k_convergence_all(results)
    plot_parameter_error(results)
    plot_information_gain(results)
    plot_point_information(results)
    plot_target_outputs(results)
    plot_designs(results)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print(f"\nSaving all outputs to: {RESULTS_DIR}\n")
    results = run_noise_study()
    export_all_outputs(results)
    plot_all(results)