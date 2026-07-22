# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:53:56 2026

@author: jahna
"""

# -*- coding: utf-8 -*-
"""
Corrected Fisher-information Bayesian OED driver for methanol production.

This is the methanol analogue of bayes_design_ROM_FIO_corrected.py.

Important methanol-specific point:
    main_meoh.jl experiments() hardcodes the true methanol parameters.
    Therefore, finite-difference sensitivities cannot be computed by passing
    perturbed parameters into experiments(). This code calls Julia main(...)
    directly with perturbed normalized k0 values, then extracts youts(...).

Information objective:
    F(x) = N_repeats / sigma^2 * J(x)^T J(x)
    info(x) = 0.5 * [logdet(Prior + F(x)) - logdet(Prior)]

where J = dY_out / dtheta, and theta is the normalized 9-parameter vector used
internally by main_meoh.jl and newton_optimizer().

Expected Julia noisy-data tensor shape:
    Y_out = (N_repeats, Nspec, Nexps)

Put this file and call_to_KPE_code_meoh_fio.py in inverse_prob_julia/.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
import concurrent.futures
import os

from call_to_KPE_code_meoh_fio import (
    experiments,
    main_model,
    parameter_estimator,
    youts,
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 6
N_REACTIONS = 2
N_UNKNOWN_PARAMETERS = 9

# Species order from main_meoh.jl:
# CO2, H2, H2O, CH3OH, CO, N2
TARGET_SPECIES_INDEX = 3

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
TRUE_PARAMS_PHYSICAL = np.array(
    [15672.02, 3453.38, 30.836, 558.532, 0.7439, 40000, 17197, 124119, 98084],
    dtype=float,
)

PARAM_LOWER = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1, 1e4, 1e4, 1e4, 1e4],
    dtype=float,
)
PARAM_UPPER = np.array(
    [1e5, 1e4, 1e5, 1e5, 1e5, 1.5e5, 1.5e5, 1.5e5, 1.5e5],
    dtype=float,
)

TRUE_PARAMS_NORMALIZED = (
    np.log10(TRUE_PARAMS_PHYSICAL) - np.log10(PARAM_LOWER)
) / (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))


def normalize_params(params_physical):
    return (
        np.log10(np.asarray(params_physical, dtype=float)) - np.log10(PARAM_LOWER)
    ) / (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))


def denormalize_params(theta):
    theta = np.asarray(theta, dtype=float)
    return 10 ** (
        theta * (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))
        + np.log10(PARAM_LOWER)
    )


INITIAL_GUESS_NORMALIZED = np.full(N_UNKNOWN_PARAMETERS, 0.1)
INITIAL_GUESS_PHYSICAL = denormalize_params(INITIAL_GUESS_NORMALIZED)

NOISE_LEVELS = [1e-5, 1e-6, 1e-7]

N_INIT = 10
MAX_EXPERIMENTS = 50
N_CANDIDATES = 50

ALLOW_EARLY_STOP_IN_SWEEP = False
CONVERGENCE_TOL = 1e-3

BASE_SEED = 12345

NOMINAL_THETA_FOR_INFORMATION = TRUE_PARAMS_NORMALIZED.copy()

FD_REL_STEP = 1e-3
FD_ABS_STEP = 1e-4
THETA_LOWER = np.zeros(N_UNKNOWN_PARAMETERS)
THETA_UPPER = np.ones(N_UNKNOWN_PARAMETERS)

FIM_NOISE_FLOOR = 1e-12

PRIOR_STD_NORMALIZED = np.ones(N_UNKNOWN_PARAMETERS)
PRIOR_PRECISION = np.diag(1.0 / (PRIOR_STD_NORMALIZED ** 2))

GP_ALPHA = 1e-8
EI_XI = 0.01
MIN_SCALED_DISTANCE = 0.05  # slightly larger to reduce flooding

RBS_FULL = False

NREF_INFORMATION = 2500
NREF_ESTIMATION = 2500


# ============================================================
# CACHES FOR DETERMINISTIC MODEL CALLS
# ============================================================

_MODEL_OUTPUT_CACHE = {}
_SENSITIVITY_CACHE = {}


def _cache_key_vector(v, decimals=10):
    return tuple(np.round(np.asarray(v, dtype=float), decimals=decimals))


# ============================================================
# PARAMETER SCALE HELPERS
# ============================================================

def clip_theta(theta):
    return np.clip(np.asarray(theta, dtype=float), THETA_LOWER, THETA_UPPER)


# ============================================================
# DESIGN HELPERS
# ============================================================

def complete_Y_in(Y_partial):
    last = 1.0 - np.sum(Y_partial)
    if last < 0.0:
        return None
    return np.append(Y_partial, last)


def decode_design_vector(x):
    x = np.asarray(x, dtype=float)
    if x.size != (NSPEC - 1) + 2:
        return None, None, None

    Y_partial = x[: NSPEC - 1]
    Temp = float(x[-2])
    P_total = float(x[-1])

    Y_full = complete_Y_in(Y_partial)
    if Y_full is None:
        return None, None, None

    return Y_full, Temp, P_total


def validate_yexp_shape(Yexp, nexps_expected, name="Yexp"):
    arr = np.asarray(Yexp, dtype=float)
    expected = (N_REPEATS, NSPEC, nexps_expected)

    if arr.shape != expected:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected}. "
            "Julia experiments() should return (N_repeats, Nspec, Nexps)."
        )

    return arr


# ============================================================
# JULIA MODEL WRAPPERS
# ============================================================

def deterministic_output(x, theta_normalized):
    x = np.asarray(x, dtype=float)
    theta_normalized = clip_theta(theta_normalized)
    key = (_cache_key_vector(x, 12), _cache_key_vector(theta_normalized, 10))

    if key in _MODEL_OUTPUT_CACHE:
        return _MODEL_OUTPUT_CACHE[key].copy()

    Y_in, Temp, P_total = decode_design_vector(x)
    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    d = main_model(
        nref=NREF_INFORMATION,
        inlet_MFs=np.asarray(Y_in, dtype=float),
        ratio=RATIO,
        St=np.array([
            [-1.0, -3.0, 1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 1.0, 0.0, 1.0, 0.0]
        ]),
        k0=np.asarray(theta_normalized, dtype=float),
        T=Temp,
        P_total=P_total,
    )

    y_out = np.asarray(youts(d, Nspec=NSPEC), dtype=float).reshape(-1)
    if y_out.size != NSPEC:
        raise ValueError(f"youts returned length {y_out.size}, expected {NSPEC}.")

    _MODEL_OUTPUT_CACHE[key] = y_out.copy()
    return y_out


def finite_difference_sensitivities(x, theta_ref):
    x = np.asarray(x, dtype=float)
    theta_ref = clip_theta(theta_ref)
    key = (
        _cache_key_vector(x, 12),
        _cache_key_vector(theta_ref, 10),
        FD_REL_STEP,
        FD_ABS_STEP,
    )

    if key in _SENSITIVITY_CACHE:
        return _SENSITIVITY_CACHE[key].copy()

    J = np.zeros((NSPEC, N_UNKNOWN_PARAMETERS), dtype=float)

    for j in range(N_UNKNOWN_PARAMETERS):
        step = max(abs(theta_ref[j]) * FD_REL_STEP, FD_ABS_STEP)
        step = min(step, 0.25)

        theta_plus = theta_ref.copy()
        theta_minus = theta_ref.copy()

        can_step_down = theta_ref[j] - step >= THETA_LOWER[j]
        can_step_up = theta_ref[j] + step <= THETA_UPPER[j]

        if can_step_down and can_step_up:
            theta_plus[j] += step
            theta_minus[j] -= step
            y_plus = deterministic_output(x, theta_plus)
            y_minus = deterministic_output(x, theta_minus)
            J[:, j] = (y_plus - y_minus) / (2.0 * step)
        elif can_step_up:
            theta_plus[j] += step
            y0 = deterministic_output(x, theta_ref)
            y_plus = deterministic_output(x, theta_plus)
            J[:, j] = (y_plus - y0) / step
        elif can_step_down:
            theta_minus[j] -= step
            y0 = deterministic_output(x, theta_ref)
            y_minus = deterministic_output(x, theta_minus)
            J[:, j] = (y0 - y_minus) / step
        else:
            raise ValueError(f"Cannot finite-difference parameter {j}.")

    _SENSITIVITY_CACHE[key] = J.copy()
    return J


def fisher_information_matrix(x, theta_ref, noise_level):
    J = finite_difference_sensitivities(x, theta_ref)
    sigma = max(float(noise_level), FIM_NOISE_FLOOR)
    weight = N_REPEATS / (sigma ** 2)
    F = weight * (J.T @ J)
    return F


def information_gain_from_F(F):
    sign0, logdet0 = np.linalg.slogdet(PRIOR_PRECISION)
    sign1, logdet1 = np.linalg.slogdet(PRIOR_PRECISION + F)

    if sign0 <= 0 or sign1 <= 0:
        return -1.0e12

    return 0.5 * (logdet1 - logdet0)


def point_information_objective(x, noise_level, theta_ref=None):
    if theta_ref is None:
        theta_ref = NOMINAL_THETA_FOR_INFORMATION

    Y_in, _, _ = decode_design_vector(x)
    if Y_in is None:
        return -1.0e12, np.zeros((N_UNKNOWN_PARAMETERS, N_UNKNOWN_PARAMETERS))

    try:
        F = fisher_information_matrix(x, theta_ref, noise_level)
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
    Y_in, Temp, P_total = decode_design_vector(x)
    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Y_in = np.asarray(Y_in, dtype=float).reshape(NSPEC, 1)
    Temp = np.asarray([Temp], dtype=float)

    Yexp = experiments(
        Y_in=Y_in,
        Temp=Temp,
        P_total=P_total,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)
    y_mean = np.mean(Yexp, axis=0)[:, 0]
    methanol = float(y_mean[TARGET_SPECIES_INDEX])

    return methanol, Yexp


def estimate_parameters_from_observations(
    X,
    Y_full,
    noise_level,
    initial_guess_normalized=None,
):
    X = np.asarray(X, dtype=float)
    Nexps = X.shape[0]

    Y_in_all = []
    Temp_all = []
    P_all = []

    for x in X:
        Y_in, Temp, P_total = decode_design_vector(x)
        if Y_in is None:
            raise ValueError(f"Invalid design vector in X: {x}")
        Y_in_all.append(Y_in)
        Temp_all.append(Temp)
        P_all.append(P_total)

    Y_in_all = np.asarray(Y_in_all, dtype=float).T
    Temp_all = np.asarray(Temp_all, dtype=float)
    P_all = np.asarray(P_all, dtype=float)

    Y_full = validate_yexp_shape(Y_full, nexps_expected=Nexps, name="Y_full")

    if initial_guess_normalized is None:
        initial_guess_normalized = INITIAL_GUESS_NORMALIZED

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
        "nref": NREF_ESTIMATION,
        "nreac": N_REACTIONS,
        "Nexps": Nexps,
        "Y_out": Y_full,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": clip_theta(initial_guess_normalized),
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
    }

    params_physical = parameter_estimator(**kwargs)
    params_physical = np.asarray(params_physical, dtype=float).reshape(-1)

    return params_physical


# ============================================================
# DESIGN GENERATION
# ============================================================

def generate_initial_design(N_init=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X_init = []

    while len(X_init) < N_init:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        Temp = rng.uniform(*TEMP_BOUNDS)
        P_total = rng.uniform(*PRESSURE_BOUNDS)
        x = np.asarray(Y_partial + [Temp, P_total], dtype=float)

        Y_full, _, _ = decode_design_vector(x)
        if Y_full is not None:
            X_init.append(x)

    return np.asarray(X_init, dtype=float)


def generate_candidates(n_candidates=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    candidates = []

    while len(candidates) < n_candidates:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        Temp = rng.uniform(*TEMP_BOUNDS)
        P_total = rng.uniform(*PRESSURE_BOUNDS)
        x = np.asarray(Y_partial + [Temp, P_total], dtype=float)

        Y_full, _, _ = decode_design_vector(x)
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
    X_scaled[:, -2] = (X[:, -2] - lb) / (ub - lb)

    lb, ub = PRESSURE_BOUNDS
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
        n_restarts_optimizer=3,
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

def relative_parameter_error(params_physical):
    params_physical = np.asarray(params_physical, dtype=float)
    denom = np.maximum(np.abs(TRUE_PARAMS_PHYSICAL), 1e-30)
    return (params_physical - TRUE_PARAMS_PHYSICAL) / denom


def check_parameter_convergence(param_history, tol=1e-3):
    if len(param_history) < 2:
        return False

    prev_theta = normalize_params(param_history[-2])
    curr_theta = normalize_params(param_history[-1])
    delta = np.linalg.norm(curr_theta - prev_theta) / np.sqrt(curr_theta.size)

    print(f"Relative normalized-parameter change: {delta:.4e}")
    return delta < tol


def BO(
    noise_level,
    N_init=5,
    max_experiments=10,
    n_candidates=200,
    allow_early_stop=True,
    tol=1e-3,
    rng_seed=None,
    initial_design=None,
):
    print("\n=== INITIAL METHANOL FIO EXPERIMENTS ===")
    print(f"noise_level = {noise_level:.0e}")

    rng = np.random.default_rng(rng_seed)

    if initial_design is None:
        X = generate_initial_design(N_init, rng=rng)
    else:
        X = np.asarray(initial_design, dtype=float).copy()

    y_info = []
    y_methanol = []
    Y_tensor_list = []
    F_matrices = []

    for i, x in enumerate(X):
        info, F = point_information_objective(
            x,
            noise_level=noise_level,
            theta_ref=NOMINAL_THETA_FOR_INFORMATION,
        )
        methanol, Yexp = run_noisy_experiment(x, noise_level=noise_level)

        y_info.append(info)
        y_methanol.append(methanol)
        F_matrices.append(F)
        Y_tensor_list.append(Yexp)

        print(f"Init Exp {i + 1}: x={x}")
        print(f"  point information = {info:.6f}")
        print(f"  methanol = {methanol:.6f}")

    y_info = np.asarray(y_info, dtype=float)
    y_methanol = np.asarray(y_methanol, dtype=float)
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=N_init,
        name="Y_full",
    )

    params_physical = estimate_parameters_from_observations(
        X,
        Y_full,
        noise_level=noise_level,
        initial_guess_normalized=INITIAL_GUESS_NORMALIZED,
    )
    param_history = [params_physical]
    param_exp_counts = [len(X)]
    total_info_history = [cumulative_information_gain(F_matrices)]

    print("Initial physical parameters =", params_physical)
    print(f"Initial cumulative information = {total_info_history[-1]:.6f}")

    print("\n=== METHANOL FIO BO START ===")

    use_parallel = True
    max_workers_override = None

    while len(X) < max_experiments:
        gp = train_gp(X, y_info)
        X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
        X_candidates = filter_far_candidates(X_candidates, X)

        ei = expected_improvement(X_candidates, gp, np.max(y_info))
        sorted_idx = np.argsort(ei)[::-1]
        top_k = 2
        selected_idx = sorted_idx[:top_k]
        X_batch = X_candidates[selected_idx]

        print(f"\n--- BO Iteration: evaluating top-{len(X_batch)} candidates ---")

        args_iterable = [(x_next, noise_level) for x_next in X_batch]

        y_new_list = []
        Yexp_new_list = []

        if use_parallel and len(X_batch) > 1:
            cpu_count = os.cpu_count() or 1
            max_workers = max_workers_override or max(1, cpu_count - 1)
            max_workers = min(max_workers, len(X_batch))

            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for methanol_val, Yexp in executor.map(
                        lambda args: run_noisy_experiment(*args),
                        args_iterable,
                    ):
                        y_new_list.append(methanol_val)
                        Yexp_new_list.append(Yexp)
                        print(f"  Candidate done: methanol={methanol_val:.6f}")
            except Exception as e:
                print("Parallel execution failed, falling back to serial. Error:", e)
                for x_next in X_batch:
                    methanol_val, Yexp = run_noisy_experiment(
                        x_next, noise_level=noise_level
                    )
                    y_new_list.append(methanol_val)
                    Yexp_new_list.append(Yexp)
                    print(f"  Candidate done (serial): methanol={methanol_val:.6f}")
        else:
            for x_next in X_batch:
                methanol_val, Yexp = run_noisy_experiment(
                    x_next, noise_level=noise_level
                )
                y_new_list.append(methanol_val)
                Yexp_new_list.append(Yexp)
                print(f"  Candidate done (serial): methanol={methanol_val:.6f}")

        info_new_list = []
        F_new_list = []
        for x_b in X_batch:
            info_b, F_b = point_information_objective(
                x_b,
                noise_level=noise_level,
                theta_ref=NOMINAL_THETA_FOR_INFORMATION,
            )
            info_new_list.append(info_b)
            F_new_list.append(F_b)

        X = np.vstack([X, X_batch])
        y_info = np.append(y_info, np.asarray(info_new_list, dtype=float))
        y_methanol = np.append(y_methanol, np.asarray(y_new_list, dtype=float))
        F_matrices.extend(F_new_list)
        Y_full = validate_yexp_shape(
            np.concatenate([Y_full] + Yexp_new_list, axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        previous_theta = clip_theta(normalize_params(param_history[-1]))
        params_physical = estimate_parameters_from_observations(
            X,
            Y_full,
            noise_level=noise_level,
            initial_guess_normalized=previous_theta,
        )

        param_history.append(params_physical)
        param_exp_counts.append(len(X))
        total_info_history.append(cumulative_information_gain(F_matrices))

        print("\n--------------------------------")
        print(f"Batch update: total experiments = {len(X)}")
        print("--------------------------------")
        print(f"Evaluated {len(X_batch)} candidates this iteration")
        print(f"Cumulative information = {total_info_history[-1]:.6f}")
        print("Estimated physical parameters =", params_physical)

        if allow_early_stop and check_parameter_convergence(param_history, tol):
            print("\nConvergence reached -> stopping early.")
            break

    return {
        "X": X,
        "point_information": y_info,
        "methanol": y_methanol,
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
        print(f"RUNNING METHANOL FIO CASE sigma = {noise:.0e}")
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

        final_params = result["param_history"][-1]
        rel_errors = relative_parameter_error(final_params)
        rel_norm = np.linalg.norm(rel_errors) / np.sqrt(N_UNKNOWN_PARAMETERS)
        best_idx = int(np.argmax(result["point_information"]))

        print("\n===== FINAL RESULTS SUMMARY =====")
        print(f"noise sigma = {noise:.0e}")
        print("Total experiments used:", len(result["X"]))
        print(f"RMS relative parameter error = {rel_norm:.6e}")
        print("Final physical parameters:")

        for i, value in enumerate(final_params):
            print(
                f"  p{i + 1}: {value:.6e} "
                f"true={TRUE_PARAMS_PHYSICAL[i]:.6e} "
                f"rel_error={rel_errors[i]:+.3e}"
            )

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


def plot_absolute_parameter_errors(results):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        label = f"sigma={_noise_label(noise)}"

        abs_errors = np.abs(params - TRUE_PARAMS_PHYSICAL.reshape(1, -1))

        for i, ax in enumerate(axes):
            ax.plot(exp_counts, abs_errors[:, i], marker="o", linewidth=1.4, label=label)

    for i, ax in enumerate(axes):
        ax.set_title(f"p{i + 1}")
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel("|error|")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True)
        _apply_plain_axis(ax)

    axes[0].legend()
    fig.suptitle("Methanol FIO Parameter Absolute Error |p - p_true|")
    fig.tight_layout()
    plt.show()


def plot_rms_absolute_parameter_error(results):
    plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        params = data["param_history"]
        exp_counts = data["param_exp_counts"]
        abs_norm = np.linalg.norm(
            params - TRUE_PARAMS_PHYSICAL.reshape(1, -1),
            axis=1,
        )
        plt.plot(
            exp_counts,
            abs_norm,
            marker="o",
            linewidth=2,
            label=f"sigma={_noise_label(noise)}",
        )

    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("Absolute parameter error (Euclidean norm)")
    plt.title("Methanol FIO Absolute Parameter Error Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
    plt.title("Methanol FIO Information Gain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_best_point_information(results):
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
    plt.title("Best Methanol FIO Pointwise Information Found by BO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_methanol_outputs(results):
    plt.figure(figsize=(9, 6))

    for noise, data in results.items():
        x_axis = np.arange(1, len(data["methanol"]) + 1)

        plt.plot(
            x_axis,
            data["methanol"],
            marker="o",
            linewidth=2,
            label=f"sigma={_noise_label(noise)}",
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Methanol outlet fraction")
    plt.title("Methanol Output at Selected FIO Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_noise_study()
    plot_absolute_parameter_errors(results)
    plot_rms_absolute_parameter_error(results)
    plot_information_gain(results)
    plot_best_point_information(results)
    plot_methanol_outputs(results)

