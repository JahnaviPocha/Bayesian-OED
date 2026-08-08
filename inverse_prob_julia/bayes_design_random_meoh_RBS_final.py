
# -*- coding: utf-8 -*-
"""
Random-design baseline for the asymptotic methanol model.

This mirrors the Julia complete_workflow settings:
- 6 species, 9 unknown parameters
- pressure is part of the design vector
- log-normalized parameter scale
- IG = TRUE_PARAMS_NORMALIZED * 0.5
- N_REPEATS = 50
- RBS_full defaults to False

It adds convergence and error plots so the random baseline can be compared
against Bayesian optimization.

Put this file in inverse_prob_julia/ next to call_to_KPE_code_meoh.py.
"""

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
try:
    from call_to_KPE_code_meoh import rpg, experiments, parameter_estimator
except ImportError:
    from call_to_KPE_code_meoh import experiments, parameter_estimator
    rpg = None


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 6
N_REACTIONS = 2
N_UNKNOWN_PARAMETERS = 9
TARGET_SPECIES_INDEX = 3  # CH3OH

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
SCALE = 1.0
N_REPEATS = 50
NOISE_LEVELS = [1e-5, 1e-6, 1e-7]

N_EXPERIMENTS = 150
MIN_ESTIMATION_EXPERIMENTS = 10

BASE_SEED = 12345
RBS_FULL = True

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_FOLDER_NAME = "bayes_design_random_meoh_RBS_final"
ADD_TIMESTAMP_TO_RESULTS_FOLDER = False

folder_name = RESULTS_FOLDER_NAME
if ADD_TIMESTAMP_TO_RESULTS_FOLDER:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{RESULTS_FOLDER_NAME}_{stamp}"

RESULTS_DIR = PROJECT_DIR / "results" / folder_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# State is written here after every estimation rather than only at the end, so
# a run that is killed or still going can still be inspected. Read it with
# tools/snapshot_progress.py.
PROGRESS_DIR = RESULTS_DIR / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


SAVE_PLOTS = True
SHOW_PLOTS = False

RUN_NOISES_IN_PARALLEL = True
MAX_PARALLEL_NOISES = min(3, len(NOISE_LEVELS))

# Random design has no BO acquisition step. These names are kept to match the
# BO scripts; here they control parallel evaluation of random experiment batches.
BATCH_SIZE = 4
CANDIDATES_IN_PARALLEL = True
MAX_CANDIDATE_WORKERS = 4

ST = np.array(
    [
        [-1.0, -3.0, 1.0, 1.0, 0.0, 0.0],
        [-1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    ],
    dtype=float,
)

TRUE_PARAMS = np.array(
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


# ============================================================
# PARAMETER SCALE
# ============================================================

def normalize_parameter_vector(params):
    params = np.asarray(params, dtype=float)
    return (np.log10(params) - np.log10(PARAM_LOWER)) / (
        np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER)
    )


def denormalize_parameter_vector(params_normalized):
    params_normalized = np.asarray(params_normalized, dtype=float)
    return 10 ** (
        params_normalized * (np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER))
        + np.log10(PARAM_LOWER)
    )


TRUE_PARAMS_NORMALIZED = normalize_parameter_vector(TRUE_PARAMS)
INITIAL_GUESS_NORMALIZED = TRUE_PARAMS_NORMALIZED * 0.5
INITIAL_GUESS_PHYSICAL = denormalize_parameter_vector(INITIAL_GUESS_NORMALIZED)


def transformed_parameter_values(values, param_index):
    """
    Display transform for parameter convergence plots.

    p1-p5: natural log of physical value.
    p6-p9: physical value, no log transform.
    This affects plotting/export display only.
    """
    values = np.asarray(values, dtype=float)

    if param_index < 5:
        if np.any(values <= 0.0):
            raise ValueError("Natural log requires positive parameter values.")
        return np.log(values), "ln(parameter value)", "Natural Log"

    return values, "Physical parameter value", "Physical"


def parameter_display_matrix(params):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)

    transformed = np.zeros_like(params, dtype=float)
    for param_index in range(N_UNKNOWN_PARAMETERS):
        transformed[:, param_index], _, _ = transformed_parameter_values(
            params[:, param_index],
            param_index,
        )

    return transformed


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
    Y_partial = x[:-2]
    Temp = float(x[-2])
    Pressure = float(x[-1])
    Y_full = complete_Y_in(Y_partial)
    if Y_full is None:
        return None, None, None
    return Y_full, Temp, Pressure


def validate_yexp_shape(Yexp, nexps_expected, name="Yexp"):
    arr = np.asarray(Yexp, dtype=float)
    expected = (N_REPEATS, NSPEC, nexps_expected)
    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")
    return arr


def design_matrix_from_julia_random(Y_in, Temp, P_total):
    Y_in = np.asarray(Y_in, dtype=float)
    Temp = np.asarray(Temp, dtype=float).reshape(-1)
    P_total = np.asarray(P_total, dtype=float).reshape(-1)
    return np.column_stack([Y_in[:-1, :].T, Temp, P_total])


def generate_random_design(n_experiments, rng):
    """
    Use Julia random_points_generator when available, otherwise Python random.
    """
    if rpg is not None:
        Y_in, Temp, P_total = rpg(
            Nexps=n_experiments,
            nspecs=NSPEC,
            lb=[0.1, 0.1, 0.0, 0.0, 0.0, 450.0, 15.0],
            ub=[0.33, 0.25, 0.01, 0.01, 0.3, 550.0, 50.0],
        )
        return design_matrix_from_julia_random(Y_in, Temp, P_total)

    X = []
    while len(X) < n_experiments:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        Temp = rng.uniform(*TEMP_BOUNDS)
        Pressure = rng.uniform(*PRESSURE_BOUNDS)
        x = np.asarray(Y_partial + [Temp, Pressure], dtype=float)
        Y_full, _, _ = decode_design_vector(x)
        if Y_full is not None:
            X.append(x)
    return np.asarray(X, dtype=float)


# ============================================================
# JULIA WRAPPERS
# ============================================================

def run_experiment(x, noise_level):
    Y_in, Temp, Pressure = decode_design_vector(x)
    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Yexp = experiments(
        scale=SCALE,
        Y_in=np.asarray(Y_in, dtype=float).reshape(NSPEC, 1),
        Temp=np.asarray([Temp], dtype=float),
        P_total=np.asarray([Pressure], dtype=float),
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)
    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    methanol = float(Y_mean[TARGET_SPECIES_INDEX])
    return methanol, Yexp


def run_experiment_worker(args):
    x, noise_level = args
    return run_experiment(np.asarray(x, dtype=float), noise_level)


def evaluate_random_experiment_batch(X_batch, noise_level):
    X_batch = np.asarray(X_batch, dtype=float)

    if (not CANDIDATES_IN_PARALLEL) or len(X_batch) <= 1:
        return [
            run_experiment(x_next, noise_level)
            for x_next in X_batch
        ]

    ctx = mp.get_context("spawn")
    max_workers = min(MAX_CANDIDATE_WORKERS, len(X_batch))
    args = [(x_next, noise_level) for x_next in X_batch]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return list(executor.map(run_experiment_worker, args))


def estimate_parameters(X, Y_full, noise_level, initial_guess=None):
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
    Y_full = validate_yexp_shape(Y_full, nexps_expected=Nexps, name="Y_full")

    if initial_guess is None:
        initial_guess = INITIAL_GUESS_NORMALIZED

    kwargs = {
        "scale": SCALE,
        "ratio": RATIO,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_all,
        "St": ST,
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

    params = parameter_estimator(**kwargs)
    return np.asarray(params, dtype=float).reshape(-1)


# ============================================================
# RANDOM STUDY
# ============================================================


def save_progress(noise_level, X, y, Y_full, param_history, param_exp_counts):
    """
    Write this noise level's state after every estimation.

    Everything else is written only once the sweep finishes, so a run that hits
    the wall clock leaves nothing behind. The noise levels are separate
    processes and each owns one file, so no locking is needed.

    The temporary name ends in .npz because savez_compressed appends .npz to
    anything that does not, which would leave the rename with no file to find.
    The rename is atomic, so a reader never sees a partial write.
    """
    path = PROGRESS_DIR / f"noise_{noise_level:.0e}.npz".replace("-", "m")
    tmp = path.with_name(path.stem + ".tmp.npz")

    np.savez_compressed(
        tmp,
        noise=noise_level,
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        Y_full=np.asarray(Y_full, dtype=float),
        params=np.asarray(param_history, dtype=float),
        param_exp_counts=np.asarray(param_exp_counts, dtype=int),
    )
    tmp.replace(path)


def random_design_study(noise_level, X_design):
    print("\n=== METHANOL RANDOM DESIGN STUDY ===")
    print(f"Noise level: {noise_level:.0e}")

    y_list = []
    Y_tensor_list = []
    param_history = []
    param_exp_counts = []
    current_guess = INITIAL_GUESS_NORMALIZED.copy()

    for batch_start in range(0, len(X_design), BATCH_SIZE):
        X_batch = X_design[batch_start:batch_start + BATCH_SIZE]
        print(
            f"\nRunning random experiment batch "
            f"{batch_start + 1}-{batch_start + len(X_batch)} "
            f"of {len(X_design)}"
        )
        batch_outputs = evaluate_random_experiment_batch(X_batch, noise_level)

        for offset, (methanol, Yexp) in enumerate(batch_outputs, start=1):
            exp_number = batch_start + offset
            y_list.append(methanol)
            Y_tensor_list.append(Yexp)
            print(
                f"Experiment {exp_number}/{len(X_design)}: "
                f"CH3OH outlet mass fraction = {methanol:.6e}"
            )

            if exp_number >= MIN_ESTIMATION_EXPERIMENTS:
                X_used = X_design[:exp_number]
                Y_full = validate_yexp_shape(
                    np.concatenate(Y_tensor_list, axis=2),
                    nexps_expected=exp_number,
                    name="Y_full",
                )
                params = estimate_parameters(
                    X_used,
                    Y_full,
                    noise_level,
                    initial_guess=current_guess,
                )
                current_guess = normalize_parameter_vector(params)
                current_guess = np.clip(current_guess, 1e-6, 0.999)
                param_history.append(params)
                param_exp_counts.append(exp_number)
                print(f"Estimated physical parameters = {params}")
                save_progress(noise_level, X_used, y_list, Y_full,
                              param_history, param_exp_counts)

    return {
        "noise": noise_level,
        "X": np.asarray(X_design, dtype=float),
        "y": np.asarray(y_list, dtype=float),
        "params": np.asarray(param_history, dtype=float),
        "param_exp_counts": np.asarray(param_exp_counts, dtype=int),
        "final_params": np.asarray(param_history[-1], dtype=float),
    }


# ============================================================
# PLOTS
# ============================================================

def noise_label(noise):
    return f"{noise:.0e}"


def apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def style_for_noise(noise):
    colors = {1e-5: "tab:blue", 1e-6: "tab:orange", 1e-7: "tab:green"}
    markers = {1e-5: "o", 1e-6: "s", 1e-7: "^"}
    linestyles = {1e-5: "-", 1e-6: "--", 1e-7: "-."}
    return {
        "color": colors.get(noise),
        "marker": markers.get(noise, "o"),
        "linestyle": linestyles.get(noise, "-"),
    }


def marker_size_for_noise(noise):
    return 9 if np.isclose(noise, 1e-5) else 7


def finish_plot(filename):
    fig = plt.gcf()

    if SAVE_PLOTS and RESULTS_DIR is not None:
        fig.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def rms_display_parameter_error(params):
    params_display = parameter_display_matrix(params)
    true_display = parameter_display_matrix(TRUE_PARAMS)[0]
    return np.linalg.norm(params_display - true_display, axis=1) / np.sqrt(
        N_UNKNOWN_PARAMETERS
    )


def build_settings_dataframe():
    return pd.DataFrame([
        {
            "N_EXPERIMENTS": N_EXPERIMENTS,
            "MIN_ESTIMATION_EXPERIMENTS": MIN_ESTIMATION_EXPERIMENTS,
            "NOISE_LEVELS": ", ".join(noise_label(noise) for noise in NOISE_LEVELS),
            "N_REPEATS": N_REPEATS,
            "RBS_FULL": RBS_FULL,
            "RUN_NOISES_IN_PARALLEL": RUN_NOISES_IN_PARALLEL,
            "MAX_PARALLEL_NOISES": MAX_PARALLEL_NOISES,
            "BATCH_SIZE": BATCH_SIZE,
            "CANDIDATES_IN_PARALLEL": CANDIDATES_IN_PARALLEL,
            "MAX_CANDIDATE_WORKERS": MAX_CANDIDATE_WORKERS,
            "BASE_SEED": BASE_SEED,
            "RESULTS_DIR": str(RESULTS_DIR),
        }
    ])


def build_summary_dataframe(results):
    rows = []
    for result in results:
        err = rms_display_parameter_error(result["final_params"])[0]
        rows.append({
            "noise": noise_label(result["noise"]),
            "n_experiments": len(result["X"]),
            "best_CH3OH": np.max(result["y"]),
            "rms_mixed_scale_parameter_error": err,
        })
    return pd.DataFrame(rows)


def build_final_parameters_dataframe(results):
    rows = []
    for result in results:
        rel_err = (result["final_params"] - TRUE_PARAMS) / np.maximum(np.abs(TRUE_PARAMS), 1e-30)
        for j in range(N_UNKNOWN_PARAMETERS):
            rows.append({
                "noise": noise_label(result["noise"]),
                "parameter": f"p{j + 1}",
                "name": PARAMETER_SHORT_NAMES[j],
                "true_value": TRUE_PARAMS[j],
                "initial_guess_physical": INITIAL_GUESS_PHYSICAL[j],
                "final_estimate": result["final_params"][j],
                "relative_error": rel_err[j],
            })
    return pd.DataFrame(rows)


def build_parameter_history_dataframe(results):
    rows = []
    true_display = parameter_display_matrix(TRUE_PARAMS)[0]
    for result in results:
        params_display = parameter_display_matrix(result["params"])
        for i, n_exp in enumerate(result["param_exp_counts"]):
            rel_err = (result["params"][i] - TRUE_PARAMS) / np.maximum(np.abs(TRUE_PARAMS), 1e-30)
            for j in range(N_UNKNOWN_PARAMETERS):
                rows.append({
                    "noise": noise_label(result["noise"]),
                    "total_experiments_used": int(n_exp),
                    "parameter": f"p{j + 1}",
                    "name": PARAMETER_SHORT_NAMES[j],
                    "estimate": result["params"][i, j],
                    "true_value": TRUE_PARAMS[j],
                    "relative_error": rel_err[j],
                    "display_estimate": params_display[i, j],
                    "display_true_value": true_display[j],
                    "display_error": params_display[i, j] - true_display[j],
                })
    return pd.DataFrame(rows)


def build_design_dataframe(X_design):
    rows = []
    for i, x in enumerate(np.asarray(X_design, dtype=float), start=1):
        rows.append({
            "experiment_number": i,
            "CO2_inlet_mass_fraction": x[0],
            "H2_inlet_mass_fraction": x[1],
            "H2O_inlet_mass_fraction": x[2],
            "CH3OH_inlet_mass_fraction": x[3],
            "CO_inlet_mass_fraction": x[4],
            "temperature_K": x[5],
            "total_pressure_bar": x[6],
        })
    return pd.DataFrame(rows)


def build_output_dataframe(results):
    rows = []
    for result in results:
        for i, y_val in enumerate(result["y"], start=1):
            rows.append({
                "noise": noise_label(result["noise"]),
                "experiment_number": i,
                "CH3OH_outlet_mass_fraction": y_val,
            })
    return pd.DataFrame(rows)


def export_results_to_excel(results, X_design):
    excel_path = RESULTS_DIR / "variable_explorer_data.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        build_settings_dataframe().to_excel(writer, sheet_name="settings", index=False)
        build_summary_dataframe(results).to_excel(writer, sheet_name="noise_summary", index=False)
        build_final_parameters_dataframe(results).to_excel(writer, sheet_name="final_parameters", index=False)
        build_parameter_history_dataframe(results).to_excel(writer, sheet_name="parameter_history", index=False)
        build_design_dataframe(X_design).to_excel(writer, sheet_name="random_design", index=False)
        build_output_dataframe(results).to_excel(writer, sheet_name="output_history", index=False)

    return excel_path


def write_summary_text(results):
    lines = ["===== METHANOL RANDOM DESIGN SUMMARY =====", ""]

    for result in results:
        lines.extend([
            "=" * 60,
            "===== FINAL RESULTS SUMMARY =====",
            "",
            f"Noise level: {result['noise']:.0e}",
            f"Total experiments used: {len(result['X'])}",
            "",
            "Final estimated parameters:",
        ])

        rel_err = (result["final_params"] - TRUE_PARAMS) / np.maximum(np.abs(TRUE_PARAMS), 1e-30)

        for j, value in enumerate(result["final_params"]):
            lines.append(
                f"  p{j + 1}: {value:.6e} "
                f"true={TRUE_PARAMS[j]:.6e} "
                f"rel_error={rel_err[j]:+.3e}"
            )

        best_idx = int(np.argmax(result["y"]))
        lines.extend([
            "",
            "Best experiment by CH3OH outlet:",
            f"  X = {np.array2string(result['X'][best_idx], precision=8)}",
            f"  CH3OH = {result['y'][best_idx]:.6e}",
            "",
        ])

    summary_path = RESULTS_DIR / "final_results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def export_all_outputs(results, X_design):
    excel_path = export_results_to_excel(results, X_design)
    summary_path = write_summary_text(results)

    print("\nSaved outputs:")
    print(f"  Folder: {RESULTS_DIR}")
    print(f"  Excel: {excel_path}")
    print(f"  Summary: {summary_path}")



def plot_design(X):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, -2], X[:, -1], c=np.arange(1, len(X) + 1), s=55)
    fig.colorbar(sc, ax=ax, label="Experiment number")
    ax.set_xlabel("CO2 inlet mass fraction")
    ax.set_ylabel("Temperature (K)")
    ax.set_zlabel("Total pressure (bar)")
    ax.set_title("Methanol Random Design Space: CO2, Temperature, Pressure")
    plt.tight_layout()
    finish_plot("01_random_design_3d.png")
    


def plot_outputs(results):
    plt.figure(figsize=(8, 5))
    for result in results:
        x_axis = np.arange(1, len(result["y"]) + 1)
        plt.plot(
            x_axis,
            result["y"],
            linewidth=1.8,
            markersize=marker_size_for_noise(result["noise"]),
            markerfacecolor="none",
            markeredgewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
            **style_for_noise(result["noise"]),
        )
    plt.xlabel("Experiment number")
    plt.ylabel("CH3OH outlet mass fraction")
    plt.title("CH3OH Outlet Fraction Across Random Experiments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    finish_plot("02_ch3oh_output.png")
    


def plot_parameter_display_scale(results):
    for j in range(N_UNKNOWN_PARAMETERS):
        plt.figure(figsize=(8, 5))
        true_value, ylabel, transform_name = transformed_parameter_values(
            [TRUE_PARAMS[j]],
            j,
        )
        initial_value, _, _ = transformed_parameter_values(
            [INITIAL_GUESS_PHYSICAL[j]],
            j,
        )

        for result in results:
            params = result["params"]
            exp_counts = result["param_exp_counts"]
            values, _, _ = transformed_parameter_values(params[:, j], j)
            plt.plot(
                exp_counts,
                values,
                linewidth=1.8,
                markersize=marker_size_for_noise(result["noise"]),
                markerfacecolor="none",
                markeredgewidth=1.6,
                label=f"noise {noise_label(result['noise'])}",
                **style_for_noise(result["noise"]),
            )
        plt.axhline(true_value[0], color="black", linestyle="--", linewidth=1.3)
        plt.axhline(initial_value[0], color="gray", linestyle=":", linewidth=1.3)
        plt.xlabel("Total experiments used")
        plt.ylabel(ylabel)
        plt.title(
            f"Methanol Random p{j + 1}: {PARAMETER_SHORT_NAMES[j]} "
            f"{transform_name} Convergence"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        finish_plot(f"03_p{j + 1:02d}_display_scale_convergence.png")
        


def plot_error(results):
    plt.figure(figsize=(8, 5))
    for result in results:
        exp_counts = result["param_exp_counts"]
        errors = rms_display_parameter_error(result["params"])
        plt.plot(
            exp_counts,
            errors,
            linewidth=1.8,
            markersize=marker_size_for_noise(result["noise"]),
            markerfacecolor="none",
            markeredgewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
            **style_for_noise(result["noise"]),
        )
    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("RMS mixed-scale parameter error")
    plt.title("Methanol Random Overall Parameter Error in Display Scale")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    finish_plot("05_rms_mixed_scale_parameter_error.png")
    


def print_summary(results):
    print("\n===== METHANOL RANDOM DESIGN SUMMARY =====")
    print("noise      n_exp    best_CH3OH      RMS_display_error")
    print("------------------------------------------------------")
    for result in results:
        err = rms_display_parameter_error(result["final_params"])[0]
        print(
            f"{result['noise']:.0e}   {len(result['X']):5d}   "
            f"{np.max(result['y']):11.4e}   {err:14.4e}"
        )


def random_design_worker(noise, X_design):
    result = random_design_study(noise, X_design)
    return noise, result


if __name__ == "__main__":
    mp.freeze_support()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving all outputs to: {RESULTS_DIR}\n")
    
    rng = np.random.default_rng(BASE_SEED)
    X_design = generate_random_design(N_EXPERIMENTS, rng)

    if RUN_NOISES_IN_PARALLEL:
        results_by_noise = {}
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=MAX_PARALLEL_NOISES,
            mp_context=ctx,
        ) as executor:
            future_to_noise = {
                executor.submit(random_design_worker, noise, X_design): noise
                for noise in NOISE_LEVELS
            }

            for future in as_completed(future_to_noise):
                noise = future_to_noise[future]
                print(f"\nCollecting finished random MeOH noise case: {noise:.0e}")
                finished_noise, result = future.result()
                results_by_noise[finished_noise] = result

        all_results = [results_by_noise[noise] for noise in NOISE_LEVELS]
    else:
        all_results = [
            random_design_study(noise, X_design)
            for noise in NOISE_LEVELS
        ]

    print_summary(all_results)
    export_all_outputs(all_results, X_design)
    
    plot_design(X_design)
    plot_outputs(all_results)
    plot_parameter_display_scale(all_results)
    plot_error(all_results)
