from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF



from call_to_KPE_code import experiments, parameter_estimator


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3
N_UNKNOWN_PARAMETERS = 2
N_REACTIONS = 1

# Design vector: x = [Y1, Y2, Temp]. Y3 is completed as 1 - Y1 - Y2.
Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

# Target species index (Python 0-based). For the ROM output, this is species 3.
TARGET_SPECIES_INDEX = 2

TRUE_K = np.array([4000.0, 4000.0], dtype=float)
INITIAL_GUESS = np.array([1000.0, 1000.0], dtype=float)

SPECIES_NAMES = ["species 1", "species 2", "species 3"]
PARAMETER_NAMES = ["k1", "k2"]

SCALE = 1.0
P_TOTAL = 50.0
RATIO = 0.1
NOISE_LEVELS = [1e-3, 1e-4, 1e-5]
STD_DATA = NOISE_LEVELS[0]  # Active noise level, updated inside the sweep.
N_REPEATS = 10
RBS_FULL = True

N_INIT = 3
MAX_EXPERIMENTS = 15
N_CANDIDATES = 200
BO_CONVERGENCE_TOL = 1e-3
ALLOW_EARLY_STOP = False

BASE_SEED = 12345
EI_XI = 0.01
GP_ALPHA_FLOOR = 1e-12


# All plots, Excel data, and text summaries are saved under this folder.
PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_FOLDER_NAME = "bayes_design_ROM_RBS"
ADD_TIMESTAMP_TO_RESULTS_FOLDER = False

folder_name = RESULTS_FOLDER_NAME
if ADD_TIMESTAMP_TO_RESULTS_FOLDER:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{RESULTS_FOLDER_NAME}_{stamp}"

RESULTS_DIR = PROJECT_DIR / "results" / folder_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = True
SHOW_PLOTS = False
PRINT_ESTIMATOR_SHAPES = True

RUN_NOISES_IN_PARALLEL = True
MAX_PARALLEL_NOISES = 3
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
    """Convert independent species mass fractions to the full 3-species inlet."""
    Y_partial = np.asarray(Y_partial, dtype=float)
    last = 1.0 - np.sum(Y_partial)

    if last < 0.0:
        return None

    return np.append(Y_partial, last)


def decode_design_vector(x):
    """Decode x = [Y1, Y2, Temp]."""
    x = np.asarray(x, dtype=float)
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
# JULIA WRAPPERS
# ============================================================

def run_experiment(x):
    """Run one Julia experiment at a physical ROM design point."""
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
        std_data=STD_DATA,
        Nspec=NSPEC,
        k_true=TRUE_K,
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)
    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    y_scalar = float(Y_mean[TARGET_SPECIES_INDEX])

    return y_scalar, Yexp


def estimate_parameters(X, Y_outputs):
    """Call Julia parameter_estimator using accumulated physical design data."""
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
    Y_out = validate_yexp_shape(Y_outputs, nexps_expected=nexps, name="Y_outputs")

    if PRINT_ESTIMATOR_SHAPES:
        print(
            "\nEstimator receives real design arrays: "
            f"Y_in={Y_in_all.shape}, "
            f"Temp={temp_all.shape}, "
            f"Y_out={Y_out.shape}"
        )

    kwargs = {
        "scale": SCALE,
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": temp_all,
        "P_total": P_TOTAL,
        "St": np.array([[-2.0, -1.0, 2.0]]),
        "nref": 2500,
        "nreac": N_REACTIONS,
        "Nexps": nexps,
        "Y_out": Y_out,
        "unknown_parameters": N_UNKNOWN_PARAMETERS,
        "IG": INITIAL_GUESS.copy(),
        "N_repeats": N_REPEATS,
        "\u03c3_data": STD_DATA,
        "RBS_full": RBS_FULL,
    }

    params = parameter_estimator(**kwargs)
    params = np.asarray(params, dtype=float).reshape(-1)

    if params.size != N_UNKNOWN_PARAMETERS:
        raise ValueError(
            f"parameter_estimator returned {params.size} values, "
            f"expected {N_UNKNOWN_PARAMETERS}."
        )

    return params


# ============================================================
# GP + EXPECTED IMPROVEMENT
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


def gp_alpha_from_noise():
    return max((STD_DATA ** 2) / max(N_REPEATS, 1), GP_ALPHA_FLOOR)


def build_gp_model(X, y):
    X_scaled = scale_X(X)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gp_alpha_from_noise(),
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


def select_next_experiment(gp, y, n_candidates=N_CANDIDATES, rng=None):
    X_candidates = generate_candidates(n_candidates=n_candidates, rng=rng)
    ei = expected_improvement(X_candidates, gp, np.max(y))
    best_idx = int(np.argmax(ei))

    return X_candidates[best_idx]


# ============================================================
# BO LOOP
# ============================================================

def relative_parameter_change(param_history):
    if len(param_history) < 2:
        return np.inf

    prev = np.asarray(param_history[-2], dtype=float)
    curr = np.asarray(param_history[-1], dtype=float)
    denom = np.maximum(np.abs(prev), np.abs(TRUE_K))
    denom = np.maximum(denom, 1e-30)

    return np.linalg.norm((curr - prev) / denom) / np.sqrt(curr.size)


def check_parameter_convergence(param_history, bo_convergence_tol=BO_CONVERGENCE_TOL):
    delta = relative_parameter_change(param_history)
    print(f"Relative parameter change: {delta:.4e}")
    return delta < bo_convergence_tol


def bayesian_optimization(
    N_init=N_INIT,
    max_experiments=MAX_EXPERIMENTS,
    bo_convergence_tol=BO_CONVERGENCE_TOL,
    n_candidates=N_CANDIDATES,
    allow_early_stop=ALLOW_EARLY_STOP,
    rng_seed=BASE_SEED,
):
    print("\n=== STARTING ROM/RBS BAYESIAN OPTIMIZATION ===\n")
    print(f"STD_DATA: {STD_DATA:.0e}")
    print(f"RBS_FULL: {RBS_FULL}")
    print(f"GP alpha: {gp_alpha_from_noise():.3e}")

    rng = np.random.default_rng(rng_seed)
    X = generate_initial_design(N_init=N_init, rng=rng)

    y_list = []
    Y_tensor_list = []

    print("\nRunning initial experiments...\n")

    for i, x in enumerate(X):
        y_scalar, Yexp = run_experiment(x)
        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)
        print(f"Init Exp {i + 1}: target output = {y_scalar:.6f}")

    y = np.asarray(y_list, dtype=float)
    Y_full = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=N_init,
        name="Y_full",
    )

    print("\nInitial parameter estimation...\n")

    params = estimate_parameters(X, Y_full)
    param_history = [params]
    param_exp_counts = [len(X)]

    print(f"Initial parameters: {params}")

    for iteration in range(max_experiments - N_init):
        print(f"\n--- BO Iteration {iteration + 1} ---")

        gp = build_gp_model(X, y)
        x_next = select_next_experiment(
            gp,
            y,
            n_candidates=n_candidates,
            rng=rng,
        )
        y_next, Yexp_next = run_experiment(x_next)

        print("New experiment:")
        print(f"  x = {x_next}")
        print(f"  target output = {y_next:.6f}")

        X = np.vstack((X, x_next))
        y = np.append(y, y_next)
        Y_full = validate_yexp_shape(
            np.concatenate((Y_full, Yexp_next), axis=2),
            nexps_expected=len(X),
            name="Y_full",
        )

        params = estimate_parameters(X, Y_full)
        param_history.append(params)
        param_exp_counts.append(len(X))

        print(f"Estimated parameters: {params}")

        if allow_early_stop and check_parameter_convergence(
            param_history,
            bo_convergence_tol,
        ):
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
# SUMMARY, EXPORT, AND PLOTS
# ============================================================

NOISE_COLORS = {
    1e-3: "tab:blue",
    1e-4: "tab:orange",
    1e-5: "tab:green",
}

NOISE_MARKERS = {
    1e-3: "o",
    1e-4: "s",
    1e-5: "^",
}

NOISE_LINESTYLES = {
    1e-3: "-",
    1e-4: "--",
    1e-5: "-.",
}


def style_for_noise(noise):
    colors = {1e-3: "tab:blue", 1e-4: "tab:orange", 1e-5: "tab:green"}
    markers = {1e-3: "o", 1e-4: "s", 1e-5: "^"}
    linestyles = {1e-3: "-", 1e-4: "--", 1e-5: "-."}

    return {
        "color": colors.get(noise),
        "marker": markers.get(noise, "o"),
        "linestyle": linestyles.get(noise, "-"),
    }

def parameter_errors(params):
    params = np.asarray(params, dtype=float)
    return params - TRUE_K


def relative_parameter_errors(params):
    params = np.asarray(params, dtype=float)
    denom = np.maximum(np.abs(TRUE_K), 1e-30)
    return (params - TRUE_K) / denom


def summarize_results(noise_level, X, y, param_history):
    print("\n===== FINAL RESULTS SUMMARY =====\n")
    print(f"Noise level: {noise_level:.0e}")
    print(f"Total experiments used: {len(X)}")

    final_params = np.asarray(param_history[-1], dtype=float)
    abs_err = parameter_errors(final_params)
    rel_err = relative_parameter_errors(final_params)

    print("\nFinal estimated parameters:")
    for i, value in enumerate(final_params):
        print(
            f"  {PARAMETER_NAMES[i]} = {value:.6f} "
            f"true={TRUE_K[i]:.6f} "
            f"error={abs_err[i]:+.6e} "
            f"rel_error={rel_err[i]:+.3e}"
        )

    best_idx = int(np.argmax(y))
    print("\nBest experiment by target output:")
    print(f"  X = {X[best_idx]}")
    print(f"  target output = {y[best_idx]:.6f}")

    return final_params


def set_active_noise(noise_level):
    global STD_DATA
    STD_DATA = float(noise_level)

def run_single_noise_case(noise_level):
    set_active_noise(noise_level)

    print("\n")
    print("=" * 60)
    print(f"RUNNING ROM/RBS BOED FOR NOISE = {noise_level:.0e}")
    print("=" * 60)

    X, y, Y_full, param_history, param_exp_counts = bayesian_optimization(
        rng_seed=BASE_SEED,
    )
    final_params = summarize_results(noise_level, X, y, param_history)

    return {
        "noise": float(noise_level),
        "X": X,
        "y": y,
        "Y_full": Y_full,
        "param_history": np.asarray(param_history, dtype=float),
        "param_exp_counts": np.asarray(param_exp_counts, dtype=int),
        "final_params": np.asarray(final_params, dtype=float),
    }

def run_noise_worker(noise_level):
    return run_single_noise_case(noise_level)

def build_settings_dataframe(results_dir):
    settings = {
        "results_dir": str(results_dir),
        "NSPEC": NSPEC,
        "N_UNKNOWN_PARAMETERS": N_UNKNOWN_PARAMETERS,
        "N_REACTIONS": N_REACTIONS,
        "TARGET_SPECIES_INDEX_python": TARGET_SPECIES_INDEX,
        "TRUE_K": ", ".join(f"{value:g}" for value in TRUE_K),
        "INITIAL_GUESS": ", ".join(f"{value:g}" for value in INITIAL_GUESS),
        "SCALE": SCALE,
        "P_TOTAL": P_TOTAL,
        "RATIO": RATIO,
        "NOISE_LEVELS": ", ".join(noise_label(noise) for noise in NOISE_LEVELS),
        "N_REPEATS": N_REPEATS,
        "RBS_FULL": RBS_FULL,
        "N_INIT": N_INIT,
        "MAX_EXPERIMENTS": MAX_EXPERIMENTS,
        "N_CANDIDATES": N_CANDIDATES,
        "BO_CONVERGENCE_TOL": BO_CONVERGENCE_TOL,
        "ALLOW_EARLY_STOP": ALLOW_EARLY_STOP,
        "BASE_SEED": BASE_SEED,
        "EI_XI": EI_XI,
        "GP_ALPHA_FLOOR": GP_ALPHA_FLOOR,
        "gp_scaled_X_passed_to_julia": False,
    }
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in settings.items()]
    )



def build_noise_sweep_summary_dataframe(all_results):
    rows = []

    for result in all_results:
        final_params = np.asarray(result["final_params"], dtype=float)
        rel_err = relative_parameter_errors(final_params)
        best_idx = int(np.argmax(result["y"]))

        rows.append(
            {
                "noise": noise_label(result["noise"]),
                "std_data": result["noise"],
                "n_experiments": len(result["X"]),
                "best_experiment_number": best_idx + 1,
                "best_target_output": float(result["y"][best_idx]),
                "final_k1": final_params[0],
                "final_k2": final_params[1],
                "true_k1": TRUE_K[0],
                "true_k2": TRUE_K[1],
                "relative_error_k1": rel_err[0],
                "relative_error_k2": rel_err[1],
                "rms_relative_parameter_error": float(
                    np.linalg.norm(rel_err) / np.sqrt(rel_err.size)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_noise_sweep_final_parameters_dataframe(all_results):
    rows = []

    for result in all_results:
        final_params = np.asarray(result["final_params"], dtype=float)
        abs_err = parameter_errors(final_params)
        rel_err = relative_parameter_errors(final_params)

        for param_index, name in enumerate(PARAMETER_NAMES):
            rows.append(
                {
                    "noise": noise_label(result["noise"]),
                    "std_data": result["noise"],
                    "parameter": name,
                    "true_value": TRUE_K[param_index],
                    "initial_guess": INITIAL_GUESS[param_index],
                    "final_estimate": final_params[param_index],
                    "absolute_error": abs_err[param_index],
                    "relative_error": rel_err[param_index],
                }
            )

    return pd.DataFrame(rows)


def build_noise_sweep_parameter_history_dataframe(all_results):
    rows = []

    for result in all_results:
        params = np.asarray(result["param_history"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)

        for row_index, exp_count in enumerate(exp_counts):
            abs_err = parameter_errors(params[row_index])
            rel_err = relative_parameter_errors(params[row_index])

            for param_index, name in enumerate(PARAMETER_NAMES):
                rows.append(
                    {
                        "noise": noise_label(result["noise"]),
                        "std_data": result["noise"],
                        "total_experiments_used": int(exp_count),
                        "parameter": name,
                        "estimate": params[row_index, param_index],
                        "true_value": TRUE_K[param_index],
                        "absolute_error": abs_err[param_index],
                        "relative_error": rel_err[param_index],
                    }
                )

    return pd.DataFrame(rows)


def build_noise_sweep_design_dataframe(all_results):
    rows = []

    for result in all_results:
        for exp_index, x in enumerate(np.asarray(result["X"], dtype=float)):
            Y_full, temp = decode_design_vector(x)
            row = {
                "noise": noise_label(result["noise"]),
                "std_data": result["noise"],
                "experiment_number": exp_index + 1,
                "target_output_species_3": result["y"][exp_index],
                "temperature_K": temp,
            }

            for species_index, species_name in enumerate(SPECIES_NAMES):
                row[f"{species_name}_inlet_mass_fraction"] = Y_full[species_index]

            rows.append(row)

    return pd.DataFrame(rows)


def build_noise_sweep_y_outputs_dataframe(all_results):
    rows = []

    for result in all_results:
        Y_full = np.asarray(result["Y_full"], dtype=float)

        for repeat_index in range(Y_full.shape[0]):
            for species_index, species_name in enumerate(SPECIES_NAMES):
                for exp_index in range(Y_full.shape[2]):
                    rows.append(
                        {
                            "noise": noise_label(result["noise"]),
                            "std_data": result["noise"],
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


def write_noise_sweep_summary_text(all_results, results_dir):
    lines = [
        "ROM/RBS MASS-KINETICS BOED NOISE SWEEP SUMMARY",
        "",
        f"Results folder: {results_dir}",
        f"Noise levels: {', '.join(noise_label(noise) for noise in NOISE_LEVELS)}",
        f"RBS_FULL: {RBS_FULL}",
        f"N_INIT: {N_INIT}",
        f"MAX_EXPERIMENTS: {MAX_EXPERIMENTS}",
        f"N_CANDIDATES: {N_CANDIDATES}",
        f"N_REPEATS: {N_REPEATS}",
        "",
    ]

    for result in all_results:
        final_params = np.asarray(result["final_params"], dtype=float)
        abs_err = parameter_errors(final_params)
        rel_err = relative_parameter_errors(final_params)
        best_idx = int(np.argmax(result["y"]))

        lines.extend(
            [
                "=" * 60,
                "===== FINAL RESULTS SUMMARY =====",
                "",
                f"Noise level: {result['noise']:.0e}",
                f"Total experiments used: {len(result['X'])}",
                "",
                "Final estimated parameters:",
            ]
        )

        for i, value in enumerate(final_params):
            lines.append(
                f"  {PARAMETER_NAMES[i]} = {value:.6f} "
                f"true={TRUE_K[i]:.6f} "
                f"error={abs_err[i]:+.6e} "
                f"rel_error={rel_err[i]:+.3e}"
            )

        lines.extend(
            [
                "",
                "Best experiment by target output:",
                f"  X = {np.array2string(np.asarray(result['X'][best_idx]), precision=8)}",
                f"  target output = {result['y'][best_idx]:.6f}",
                "",
            ]
        )

    summary_path = results_dir / "final_results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def export_noise_sweep_outputs(all_results, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    excel_path = results_dir / "variable_explorer_data.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        build_settings_dataframe(results_dir).to_excel(
            writer,
            sheet_name="settings",
            index=False,
        )
        build_noise_sweep_summary_dataframe(all_results).to_excel(
            writer,
            sheet_name="noise_summary",
            index=False,
        )
        build_noise_sweep_final_parameters_dataframe(all_results).to_excel(
            writer,
            sheet_name="final_parameters",
            index=False,
        )
        build_noise_sweep_parameter_history_dataframe(all_results).to_excel(
            writer,
            sheet_name="parameter_history",
            index=False,
        )
        build_noise_sweep_design_dataframe(all_results).to_excel(
            writer,
            sheet_name="selected_designs",
            index=False,
        )
        build_noise_sweep_y_outputs_dataframe(all_results).to_excel(
            writer,
            sheet_name="Y_outputs_flat",
            index=False,
        )

    summary_path = write_noise_sweep_summary_text(all_results, results_dir)

    print("\nSaved combined noise-sweep outputs:")
    print(f"  Excel data: {excel_path}")
    print(f"  Summary: {summary_path}")


def plot_experimental_points_distribution(all_results, results_dir=None):
    fig, ax = plt.subplots(figsize=(9, 6))

    for result in all_results:
        X = np.asarray(result["X"], dtype=float)
        style = style_for_noise(result["noise"])

        ax.scatter(
            X[:, 0],
            X[:, -1],
            s=90,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.7,
            label=f"noise {noise_label(result['noise'])}",
            marker=style["marker"],
            color=style["color"],
        )
        ax.plot(
            X[:, 0],
            X[:, -1],
            alpha=0.45,
            linewidth=1.2,
            linestyle=style["linestyle"],
            color=style["color"],
        )

    ax.set_xlabel("Y1 inlet mass fraction")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Experimental Points Distribution Across Noise Levels")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "01_experimental_points_distribution.png")


def plot_noise_sweep_parameter_convergence(all_results, results_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for result in all_results:
        params = np.asarray(result["param_history"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])

        for param_index, ax in enumerate(axes):
            ax.plot(
                exp_counts,
                params[:, param_index],
                linewidth=1.8,
                markersize=7,
                label=label,
                **style,
            )

    for param_index, ax in enumerate(axes):
        ax.axhline(TRUE_K[param_index], color="black", linestyle="--", linewidth=1.2)
        ax.axhline(INITIAL_GUESS[param_index], color="gray", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"Estimated {PARAMETER_NAMES[param_index]}")
        ax.set_title(f"{PARAMETER_NAMES[param_index]} Convergence")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save_and_show(fig, results_dir, "02_parameter_convergence.png")


def plot_noise_sweep_parameter_errors(all_results, results_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for result in all_results:
        params = np.asarray(result["param_history"], dtype=float)
        exp_counts = np.asarray(result["param_exp_counts"], dtype=int)
        rel_err = relative_parameter_errors(params)
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])

        for param_index, ax in enumerate(axes):
            ax.plot(
                exp_counts,
                rel_err[:, param_index],
                linewidth=1.8,
                markersize=7,
                label=label,
                **style,
            )

    for param_index, ax in enumerate(axes):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel("(estimate - true) / |true|")
        ax.set_title(f"{PARAMETER_NAMES[param_index]} Relative Error")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save_and_show(fig, results_dir, "03_parameter_relative_errors.png")


def plot_noise_sweep_target_outputs(all_results, results_dir=None):
    fig = plt.figure(figsize=(9, 5))

    for result in all_results:
        exp_numbers = np.arange(1, len(result["y"]) + 1)
        plt.plot(
            exp_numbers,
            result["y"],
            linewidth=1.8,
            markersize=7,
            label=f"noise {noise_label(result['noise'])}",
            **style_for_noise(result["noise"]),
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Target output fraction")
    plt.title("Target Output Across Noise Levels")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, results_dir, "04_target_outputs.png")


def plot_noise_sweep_results(all_results, results_dir=None):
    plot_experimental_points_distribution(all_results, results_dir)
    plot_noise_sweep_parameter_convergence(all_results, results_dir)
    plot_noise_sweep_parameter_errors(all_results, results_dir)
    plot_noise_sweep_target_outputs(all_results, results_dir)


# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()
    all_results = []
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving all outputs to: {results_dir}\n")
    
    if RUN_NOISES_IN_PARALLEL:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=MAX_PARALLEL_NOISES,
            mp_context=ctx,
        ) as executor:
            future_to_noise = {
                executor.submit(run_noise_worker, noise): noise
                for noise in NOISE_LEVELS
            }

            for future in as_completed(future_to_noise):
                noise = future_to_noise[future]
                print(f"\nCollecting finished noise case: {noise:.0e}")
                all_results.append(future.result())

        all_results.sort(key=lambda result: NOISE_LEVELS.index(result["noise"]))

    else:

        for noise_level in NOISE_LEVELS:
            all_results.append(run_single_noise_case(noise_level))

    export_noise_sweep_outputs(all_results, results_dir)
    plot_noise_sweep_results(all_results, results_dir)