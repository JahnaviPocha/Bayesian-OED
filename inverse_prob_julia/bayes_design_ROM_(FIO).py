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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

from call_to_KPE_code import experiments, parameter_estimator
import concurrent.futures
import os


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3
N_REACTIONS = 1
N_UNKNOWN_PARAMETERS = 2

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

TARGET_SPECIES_INDEX = 2

TRUE_K = np.array([4000.0, 4000.0], dtype=float)
INITIAL_GUESS = np.array([1000.0, 1000.0], dtype=float)

P_TOTAL = 50
RATIO = 0.1
STOICHIOMETRY = np.array([[-2.0, -1.0, 2.0]])

N_REPEATS = 10
NOISE_LEVELS = [1e-3, 1e-5, 1e-7]

N_INIT = 3
MAX_EXPERIMENTS = 10
N_CANDIDATES = 200

# Keep False for fair noise-comparison plots. Set True for faster single runs.
ALLOW_EARLY_STOP_IN_SWEEP = False
CONVERGENCE_TOL = 1e-3

BASE_SEED = 12345

# This is a local D-optimal design. In synthetic validation it is common to use
# the true parameter as the nominal value. For real experiments, replace this
# with your best prior/previous estimate.
NOMINAL_K_FOR_INFORMATION = TRUE_K.copy()

FD_REL_STEP = 1e-3
FD_ABS_STEP = 1e-3

# Avoid unrealistically infinite information when sigma is extremely tiny.
FIM_NOISE_FLOOR = 1e-12

# Weak prior precision keeps logdet well-defined before adding information.
PRIOR_STD = np.array([1.0e4, 1.0e4], dtype=float)
PRIOR_PRECISION = np.diag(1.0 / (PRIOR_STD ** 2))

GP_ALPHA = 1e-8
EI_XI = 0.01
MIN_SCALED_DISTANCE = 0.03

RBS_FULL = True


# ============================================================
# CACHES FOR DETERMINISTIC MODEL CALLS
# ============================================================

_MODEL_OUTPUT_CACHE = {}
_SENSITIVITY_CACHE = {}


def _cache_key_vector(v, decimals=10):
    return tuple(np.round(np.asarray(v, dtype=float), decimals=decimals))


# ============================================================
# DESIGN HELPERS
# ============================================================

def complete_Y_in(Y_partial):
    """Convert independent inlet mass fractions to the full species vector."""
    last = 1.0 - np.sum(Y_partial)

    if last < 0.0:
        return None

    return np.append(Y_partial, last)


def decode_design_vector(x):
    """Decode x = [Y1, Y2, Temp]."""
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

    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Y_in = np.asarray(Y_in, dtype=float).reshape(NSPEC, 1)
    Temp = np.asarray([Temp], dtype=float)

    Yexp = experiments(
        Y_in=Y_in,
        Temp=Temp,
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
    """
    Compute J = dY_out / dk at one design using finite differences.

    J shape:
        (Nspec, N_UNKNOWN_PARAMETERS)
    """
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
    """
    Local Fisher information matrix for one candidate experiment.

    Assumes all species outputs have independent Gaussian noise with standard
    deviation noise_level, and N_REPEATS repeated measurements are averaged.
    """
    J = finite_difference_sensitivities(x, k_ref)
    sigma = max(float(noise_level), FIM_NOISE_FLOOR)
    weight = N_REPEATS / (sigma ** 2)
    F = weight * (J.T @ J)

    return F


def information_gain_from_F(F):
    """D-optimal information gain relative to the weak prior precision."""
    sign0, logdet0 = np.linalg.slogdet(PRIOR_PRECISION)
    sign1, logdet1 = np.linalg.slogdet(PRIOR_PRECISION + F)

    if sign0 <= 0 or sign1 <= 0:
        return -1.0e12

    return 0.5 * (logdet1 - logdet0)


def point_information_objective(x, noise_level, k_ref=None):
    """
    Pointwise D-optimal information score for one experiment.

    The BO surrogate learns this pointwise score. The selected design's
    cumulative information is tracked separately from the sum of F matrices.
    """
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
    """D-optimal information gain for the full selected design."""
    if not F_matrices:
        return 0.0

    F_total = np.sum(np.asarray(F_matrices, dtype=float), axis=0)

    return information_gain_from_F(F_total)


def run_noisy_experiment(x, noise_level):
    """
    Run one noisy experiment used for parameter estimation.

    Returns:
        target scalar and full Yexp tensor with shape (N_repeats, Nspec, 1)
    """
    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

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
    Y_full = validate_yexp_shape(Y_full, nexps_expected=Nexps, name="Y_full")

    if initial_guess is None:
        initial_guess = INITIAL_GUESS

    kwargs = {
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_TOTAL,
        "St": STOICHIOMETRY,
        "nref": 2500,
        "nreac": N_REACTIONS,
        "Nexps": Nexps,
        "Y_out": Y_full,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": np.asarray(initial_guess, dtype=float),
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
    }

    k = parameter_estimator(**kwargs)
    k = np.asarray(k, dtype=float).reshape(-1)

    return k


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design(N_init=3, rng=None):
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
# GP SURROGATE + EXPECTED IMPROVEMENT
# ============================================================

def scale_X(X):
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
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
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


def select_next_experiment(gp, X, y_info, rng=None, n_candidates=500):
    X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
    X_candidates = filter_far_candidates(X_candidates, X)

    ei = expected_improvement(X_candidates, gp, np.max(y_info))
    best_idx = int(np.argmax(ei))

    return X_candidates[best_idx]


# ============================================================
# BO LOOP
# ============================================================

def check_parameter_convergence(param_history, tol=1e-3):
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
    N_init=3,
    max_experiments=10,
    n_candidates=500,
    allow_early_stop=True,
    tol=1e-3,
    rng_seed=None,
    initial_design=None,
):
    print("\n=== INITIAL EXPERIMENTS ===")
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
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=N_init,
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

    print("\n=== BO START ===")

        # --- inside BO, replace the while-loop body with this ---
    use_parallel = True  # set False if Julia bridge is not process-safe
    max_workers_override = None  # set to an int to limit workers, else None

    while len(X) < max_experiments:
        gp = train_gp(X, y_info)
        X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
        X_candidates = filter_far_candidates(X_candidates, X)

        ei = expected_improvement(X_candidates, gp, np.max(y_info))
        sorted_idx = np.argsort(ei)[::-1]
        X_batch = X_candidates[sorted_idx]

        # Optionally limit to top-k candidates to reduce cost:
        # k = 20
        # X_batch = X_batch[:k]

        print(f"\n--- BO Iteration: selecting {len(X_batch)} candidates to evaluate ---")

        args_iterable = [(x_next, noise_level) for x_next in X_batch]

        y_new_list = []
        Yexp_new_list = []

        if use_parallel and len(X_batch) > 1:
            cpu_count = os.cpu_count() or 1
            max_workers = max_workers_override or max(1, cpu_count - 1)
            max_workers = min(max_workers, len(X_batch))

            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # executor.map keeps order consistent with X_batch
                    for y_next, Yexp_next in executor.map(lambda args: run_noisy_experiment(*args), args_iterable):
                        y_new_list.append(y_next)
                        Yexp_new_list.append(Yexp_next)
                        print(f"  Candidate done: target={y_next:.6f}")
            except Exception as e:
                # Fallback to serial if parallel execution fails
                print("Parallel execution failed, falling back to serial. Error:", e)
                for x_next in X_batch:
                    y_next, Yexp_next = run_noisy_experiment(x_next, noise_level=noise_level)
                    y_new_list.append(y_next)
                    Yexp_new_list.append(Yexp_next)
                    print(f"  Candidate done (serial): target={y_next:.6f}")
        else:
            for x_next in X_batch:
                y_next, Yexp_next = run_noisy_experiment(x_next, noise_level=noise_level)
                y_new_list.append(y_next)
                Yexp_new_list.append(Yexp_next)
                print(f"  Candidate done (serial): target={y_next:.6f}")

        # Update data once per iteration
        X = np.vstack([X, X_batch])
        y_info = np.append(y_info, np.asarray([point_information_objective(x, noise_level)[0] for x in X_batch]))
        y_target = np.append(y_target, y_new_list)
        F_matrices.extend([point_information_objective(x, noise_level)[1] for x in X_batch])
        Y_full = validate_yexp_shape(
            np.concatenate([Y_full] + Yexp_new_list, axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        # Parameter estimation (warm-start)
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
        print(f"Experiment {len(X)} (batch update)")
        print("--------------------------------")
        print(f"New designs evaluated: {len(X_batch)}")
        print(f"Cumulative information = {total_info_history[-1]:.6f}")
        print(f"Target outputs (last batch): {y_new_list[:5]} ...")
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


# ============================================================
# NOISE STUDY
# ============================================================

def run_noise_study():
    results = {}

    initial_rng = np.random.default_rng(BASE_SEED)
    shared_initial_design = generate_initial_design(N_INIT, rng=initial_rng)

    for noise in NOISE_LEVELS:
        print("\n")
        print("=" * 60)
        print(f"RUNNING NOISE CASE sigma = {noise:.0e}")
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
        best_idx = int(np.argmax(result["point_information"]))
        rel_error = np.linalg.norm(final_k - TRUE_K) / np.linalg.norm(TRUE_K)

        print("\n===== FINAL RESULTS SUMMARY =====")
        print(f"noise sigma = {noise:.0e}")
        print("Total experiments used:", len(result["X"]))
        print(f"k1 = {final_k[0]:.6f}")
        print(f"k2 = {final_k[1]:.6f}")
        print(f"Relative parameter error = {rel_error:.6e}")
        print("Best pointwise-information experiment:")
        print(result["X"][best_idx])
        print(f"Point information = {result['point_information'][best_idx]:.6f}")
        print(f"Final cumulative information = {result['total_info_history'][-1]:.6f}")

        results[noise] = result

    return results


# ============================================================
# PLOTS
# ============================================================

def _noise_label(noise):
    return f"{noise:.0e}"


def _apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_k_convergence_all(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        label = f"sigma={_noise_label(noise)}"

        axes[0].plot(exp_counts, params[:, 0], marker="o", linewidth=2, label=label)
        axes[1].plot(exp_counts, params[:, 1], marker="s", linewidth=2, label=label)

    for i, ax in enumerate(axes):
        ax.axhline(TRUE_K[i], color="black", linestyle="--", linewidth=2)
        ax.axhline(INITIAL_GUESS[i], color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"Estimated k{i + 1}")
        ax.set_title(f"k{i + 1} convergence")
        ax.grid(True)
        _apply_plain_axis(ax)

    axes[0].legend()
    fig.tight_layout()
    plt.show()


def plot_parameter_error(results):
    # 1) Norm of absolute error (combined)
    plt.figure(figsize=(9, 5))
    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        abs_norm = np.linalg.norm(params - TRUE_K, axis=1)  # Euclidean absolute error
        plt.plot(exp_counts, abs_norm, marker="o", linewidth=2, label=f"sigma={_noise_label(noise)}")

    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("Absolute parameter error (Euclidean norm)")
    plt.title("Absolute Parameter Error (norm) Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Per-parameter absolute errors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        abs_err_k1 = np.abs(params[:, 0] - TRUE_K[0])
        abs_err_k2 = np.abs(params[:, 1] - TRUE_K[1])

        axes[0].plot(exp_counts, abs_err_k1, marker="o", linewidth=2, label=f"sigma={_noise_label(noise)}")
        axes[1].plot(exp_counts, abs_err_k2, marker="s", linewidth=2, label=f"sigma={_noise_label(noise)}")

    for i, ax in enumerate(axes):
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"|error in k{i + 1}|")
        ax.set_title(f"Absolute Parameter Error: k{i + 1}")
        ax.grid(True)
        # show both small and large errors
        ax.set_yscale("symlog", linthresh=1e-6)
        _apply_plain_axis(ax)

    axes[0].legend()
    fig.tight_layout()
    plt.show()



def plot_information_gain(results):
    plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        plt.plot(
            data["param_exp_counts"],
            data["total_info_history"],
            marker="o",
            linewidth=2,
            label=f"sigma={_noise_label(noise)}",
        )

    plt.xlabel("Total experiments used")
    plt.ylabel("Cumulative D-optimal information gain")
    plt.title("Bayesian OED Information Gain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_point_information(results):
    plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        x_axis = np.arange(1, len(data["point_information"]) + 1)
        best_info = np.maximum.accumulate(data["point_information"])

        plt.plot(
            x_axis,
            best_info,
            marker="o",
            linewidth=2,
            label=f"sigma={_noise_label(noise)}",
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Best pointwise information")
    plt.title("Best Pointwise Information Found by BO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_designs(results):
    plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        X = data["X"]
        plt.plot(
            X[:, 0],
            X[:, -1],
            marker="o",
            linewidth=1.5,
            label=f"sigma={_noise_label(noise)}",
        )

    plt.xlabel("Y1")
    plt.ylabel("Temperature (K)")
    plt.title("Selected Experimental Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all(results):
    plot_k_convergence_all(results)
    plot_parameter_error(results)
    plot_information_gain(results)
    plot_point_information(results)
    plot_designs(results)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    results = run_noise_study()
    plot_all(results)
