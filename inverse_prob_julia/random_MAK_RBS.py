# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:50:53 2026

@author: jahna
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

P_TOTAL = 50
RATIO = 0.1
ST = np.array([[-2.0, -1.0, 2.0]])
SCALE = 1.0

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_FOLDER_NAME = "bayes_design_random_ROM_asymptotic"
ADD_TIMESTAMP_TO_RESULTS_FOLDER = False

folder_name = RESULTS_FOLDER_NAME
if ADD_TIMESTAMP_TO_RESULTS_FOLDER:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{RESULTS_FOLDER_NAME}_{stamp}"

RESULTS_DIR = PROJECT_DIR / "results" / folder_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = True
SHOW_PLOTS = False

N_REPEATS = 10
NOISE_LEVELS = [1e-3, 1e-4, 1e-5]

N_EXPERIMENTS = 15
MIN_ESTIMATION_EXPERIMENTS = 3

BASE_SEED = 12345
RBS_FULL = True

RUN_NOISES_IN_PARALLEL = True
MAX_PARALLEL_NOISES = min(3, len(NOISE_LEVELS))

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
    Y_partial = x[:-1]
    Temp = float(x[-1])
    Y_full = complete_Y_in(Y_partial)
    if Y_full is None:
        return None, None
    return Y_full, Temp


def validate_yexp_shape(Yexp, nexps_expected, name="Yexp"):
    arr = np.asarray(Yexp, dtype=float)
    expected = (N_REPEATS, NSPEC, nexps_expected)
    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")
    return arr


# ============================================================
# JULIA WRAPPERS
# ============================================================

def run_experiment(x, noise_level):
    Y_in, Temp = decode_design_vector(x)
    if Y_in is None:
        raise ValueError(f"Invalid design vector: {x}")

    Yexp = experiments(
        scale=SCALE,
        Y_in=np.asarray(Y_in, dtype=float).reshape(NSPEC, 1),
        Temp=np.asarray([Temp], dtype=float),
        P_total=P_TOTAL,
        Nexps=1,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=noise_level,
        Nspec=NSPEC,
        k_true=TRUE_K.tolist(),
    )

    Yexp = validate_yexp_shape(Yexp, nexps_expected=1)
    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    y_scalar = float(Y_mean[TARGET_SPECIES_INDEX])
    return y_scalar, Yexp


def estimate_parameters(X, Y_full, noise_level, initial_guess=None):
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
        "scale": SCALE,
        "nspec": NSPEC,
        "Y_in": Y_in_all,
        "Temp": Temp_all,
        "P_total": P_TOTAL,
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
# RANDOM DESIGN
# ============================================================

def generate_random_design(n_experiments, rng):
    X = []
    while len(X) < n_experiments:
        Y_partial = [rng.uniform(*Y_BOUNDS[i]) for i in range(NSPEC - 1)]
        Temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [Temp], dtype=float)
        Y_full, _ = decode_design_vector(x)
        if Y_full is not None:
            X.append(x)
    return np.asarray(X, dtype=float)


def random_design_study(noise_level, X_design):
    print("\n=== ROM RANDOM DESIGN STUDY ===")
    print(f"Noise level: {noise_level:.0e}")

    y_list = []
    Y_tensor_list = []
    param_history = []
    param_exp_counts = []
    current_guess = INITIAL_GUESS.copy()

    for exp_number, x in enumerate(X_design, start=1):
        print(f"\nExperiment {exp_number}/{len(X_design)}")
        y_scalar, Yexp = run_experiment(x, noise_level)
        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)
        print(f"Target output = {y_scalar:.6f}")

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
            current_guess = params
            param_history.append(params)
            param_exp_counts.append(exp_number)
            print(f"Estimated k = {params}")

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


def style_for_noise(noise):
    colors = {
        1e-3: "tab:blue",
        1e-4: "tab:orange",
        1e-5: "tab:green",
    }
    markers = {
        1e-3: "o",
        1e-4: "s",
        1e-5: "^",
    }
    linestyles = {
        1e-3: "-",
        1e-4: "--",
        1e-5: "-.",
    }
    return {
        "color": colors.get(noise),
        "marker": markers.get(noise, "o"),
        "linestyle": linestyles.get(noise, "-"),
    }


def apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def finish_plot(filename):
    fig = plt.gcf()

    if SAVE_PLOTS and RESULTS_DIR is not None:
        fig.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def build_summary_dataframe(results):
    rows = []
    for result in results:
        params = result["final_params"]
        rel_error = np.linalg.norm(params - TRUE_K) / np.linalg.norm(TRUE_K)

        rows.append({
            "noise": noise_label(result["noise"]),
            "n_experiments": len(result["X"]),
            "best_target_output": np.max(result["y"]),
            "k1_final": params[0],
            "k2_final": params[1],
            "k1_true": TRUE_K[0],
            "k2_true": TRUE_K[1],
            "relative_parameter_error": rel_error,
        })

    return pd.DataFrame(rows)


def build_final_parameters_dataframe(results):
    rows = []
    for result in results:
        params = result["final_params"]
        abs_err = params - TRUE_K
        rel_err = abs_err / np.maximum(np.abs(TRUE_K), 1e-30)

        for j in range(N_UNKNOWN_PARAMETERS):
            rows.append({
                "noise": noise_label(result["noise"]),
                "parameter": f"k{j + 1}",
                "true_value": TRUE_K[j],
                "initial_guess": INITIAL_GUESS[j],
                "final_estimate": params[j],
                "absolute_error": abs_err[j],
                "relative_error": rel_err[j],
            })

    return pd.DataFrame(rows)


def build_parameter_history_dataframe(results):
    rows = []
    for result in results:
        for i, n_exp in enumerate(result["param_exp_counts"]):
            params = result["params"][i]
            abs_err = params - TRUE_K
            rel_err = abs_err / np.maximum(np.abs(TRUE_K), 1e-30)

            for j in range(N_UNKNOWN_PARAMETERS):
                rows.append({
                    "noise": noise_label(result["noise"]),
                    "total_experiments_used": int(n_exp),
                    "parameter": f"k{j + 1}",
                    "estimate": params[j],
                    "true_value": TRUE_K[j],
                    "absolute_error": abs_err[j],
                    "relative_error": rel_err[j],
                })

    return pd.DataFrame(rows)


def build_selected_designs_dataframe(X_design):
    rows = []
    for i, x in enumerate(X_design):
        Y_full, Temp = decode_design_vector(x)

        rows.append({
            "experiment_number": i + 1,
            "Y1_inlet_mass_fraction": Y_full[0],
            "Y2_inlet_mass_fraction": Y_full[1],
            "Y3_inlet_mass_fraction": Y_full[2],
            "temperature_K": Temp,
        })

    return pd.DataFrame(rows)


def write_summary_text(results):
    lines = ["===== ROM RANDOM DESIGN SUMMARY =====", ""]

    for result in results:
        params = result["final_params"]
        abs_err = params - TRUE_K
        rel_err = abs_err / np.maximum(np.abs(TRUE_K), 1e-30)
        best_idx = int(np.argmax(result["y"]))

        lines.extend([
            "=" * 60,
            "===== FINAL RESULTS SUMMARY =====",
            "",
            f"Noise level: {result['noise']:.0e}",
            f"Total experiments used: {len(result['X'])}",
            "",
            "Final estimated parameters:",
        ])

        for j, value in enumerate(params):
            lines.append(
                f"  k{j + 1}: {value:.6f} "
                f"true={TRUE_K[j]:.6f} "
                f"error={abs_err[j]:+.6e} "
                f"rel_error={rel_err[j]:+.3e}"
            )

        lines.extend([
            "",
            "Best experiment by target output:",
            f"  X = {np.array2string(result['X'][best_idx], precision=8)}",
            f"  target output = {result['y'][best_idx]:.6f}",
            "",
        ])

    summary_path = RESULTS_DIR / "final_results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path        
        
def export_results_to_excel(results, X_design):
    excel_path = RESULTS_DIR / "variable_explorer_data.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        build_summary_dataframe(results).to_excel(writer, sheet_name="noise_summary", index=False)
        build_final_parameters_dataframe(results).to_excel(writer, sheet_name="final_parameters", index=False)
        build_parameter_history_dataframe(results).to_excel(writer, sheet_name="parameter_history", index=False)
        build_selected_designs_dataframe(X_design).to_excel(writer, sheet_name="selected_designs", index=False)

    return excel_path




def export_all_outputs(results, X_design):
    excel_path = export_results_to_excel(results, X_design)
    summary_path = write_summary_text(results)

    print("\nSaved outputs:")
    print(f"  Folder: {RESULTS_DIR}")
    print(f"  Excel: {excel_path}")
    print(f"  Summary: {summary_path}")
        

def plot_design(X):
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, -1], c=np.arange(1, len(X) + 1), s=90)
    plt.colorbar(label="Experiment number")
    plt.xlabel("Y1 inlet mass fraction")
    plt.ylabel("Temperature (K)")
    plt.title("ROM Random Experimental Design")
    plt.grid(True)
    plt.tight_layout()
    finish_plot("01_random_design.png")
    


def plot_outputs(results):
    plt.figure(figsize=(8, 5))
    for result in results:
        x_axis = np.arange(1, len(result["y"]) + 1)
        plt.plot(
            x_axis,
            result["y"],
            **style_for_noise(result["noise"]),
            linewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
        )
    plt.xlabel("Experiment number")
    plt.ylabel("Target species outlet fraction")
    plt.title("ROM Random Design Output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    finish_plot("02_target_output.png")
   


def plot_parameter_convergence(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for result in results:
        params = result["params"]
        exp_counts = result["param_exp_counts"]
        label = f"noise {noise_label(result['noise'])}"
        style = style_for_noise(result["noise"])
        axes[0].plot(exp_counts, params[:, 0], **style, linewidth=1.6, label=label)
        axes[1].plot(exp_counts, params[:, 1], **style, linewidth=1.6, label=label)


    for i, ax in enumerate(axes):
        ax.axhline(TRUE_K[i], color="black", linestyle="--", linewidth=1.5)
        ax.axhline(INITIAL_GUESS[i], color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"k{i + 1}")
        ax.set_title(f"ROM Random Design k{i + 1} Convergence")
        ax.grid(True)
        apply_plain_axis(ax)
    axes[0].legend()
    fig.tight_layout()
    finish_plot("03_parameter_convergence.png")
    


def plot_parameter_error(results):
    plt.figure(figsize=(8, 5))
    for result in results:
        params = result["params"]
        exp_counts = result["param_exp_counts"]
        errors = np.linalg.norm(params - TRUE_K, axis=1) / np.linalg.norm(TRUE_K)
        plt.plot(
            exp_counts,
            errors,
            **style_for_noise(result["noise"]),
            linewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
        )
    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("Relative parameter error")
    plt.title("ROM Random Design Parameter Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    finish_plot("04_parameter_error.png")
   

def print_summary(results):
    print("\n===== ROM RANDOM DESIGN SUMMARY =====")
    print("noise      n_exp    k1_final       k2_final       rel_error")
    print("-------------------------------------------------------------")
    for result in results:
        params = result["final_params"]
        err = np.linalg.norm(params - TRUE_K) / np.linalg.norm(TRUE_K)
        print(
            f"{result['noise']:.0e}   {len(result['X']):5d}   "
            f"{params[0]:12.6f}   {params[1]:12.6f}   {err:10.3e}"
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
                print(f"\nCollecting finished random noise case: {noise:.0e}")
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
    plot_parameter_convergence(all_results)
    plot_parameter_error(all_results)