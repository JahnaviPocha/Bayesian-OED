# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:48:52 2026

@author: jahna
"""

"""
Random-design baseline with parameter convergence/error plots.

Use this as a baseline against the Bayesian optimization runs:
- same random design is reused for all noise levels
- observations are accumulated experiment by experiment
- parameter_estimator is called on the cumulative data
- plots show k1/k2 convergence and relative parameter error

Put this file in inverse_prob_julia/ next to call_to_KPE_code.py before running.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from call_to_KPE_code import experiments, parameter_estimator


# ============================================================
# GLOBAL SETTINGS
# ============================================================

NSPEC = 3
TARGET_SPECIES_INDEX = 2

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

TRUE_K = np.array([4000.0, 4000.0], dtype=float)
INITIAL_GUESS = np.array([1000.0, 1000.0], dtype=float)

P_TOTAL = 50
RATIO = 0.1
N_REPEATS = 10

NOISE_LEVELS = [1e-3, 1e-4, 1e-5]

N_EXPERIMENTS = 8
MIN_ESTIMATION_EXPERIMENTS = 3

BASE_SEED = 12345
RBS_FULL = True


# ============================================================
# DESIGN HELPERS
# ============================================================

def complete_Y_in(Y_partial):
    """Convert independent mass fractions to full inlet vector."""
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
    """Check Julia tensor shape: (N_repeats, Nspec, Nexps)."""
    arr = np.asarray(Yexp, dtype=float)
    expected = (N_REPEATS, NSPEC, nexps_expected)

    if arr.shape != expected:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected}. "
            "Julia experiments() should return (N_repeats, Nspec, Nexps)."
        )

    return arr


# ============================================================
# JULIA WRAPPERS
# ============================================================

def run_experiment(x, noise_level):
    """
    Run one random-design experiment through Julia.

    Returns:
        y_scalar: mean target species output
        Yexp: full noisy tensor with shape (N_repeats, Nspec, 1)
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
    Y_mean = np.mean(Yexp, axis=0)[:, 0]
    y_scalar = float(Y_mean[TARGET_SPECIES_INDEX])

    return y_scalar, Yexp


def estimate_parameters(X, Y_full, noise_level, initial_guess=None):
    """Estimate parameters from accumulated random-design observations."""
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

    initial_guess = np.clip(np.asarray(initial_guess, dtype=float), 0.0, 20000.0)

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
        "Y_out": Y_full,
        "unknown_parameters": 2,
        "IG": initial_guess,
        "N_repeats": N_REPEATS,
        "\u03c3_data": noise_level,
        "RBS_full": RBS_FULL,
    }

    params = parameter_estimator(**kwargs)

    return np.asarray(params, dtype=float).reshape(-1)



# ============================================================
# RANDOM DESIGN
# ============================================================

def generate_random_design(N_experiments=8, rng=None):
    """Generate a valid random design."""
    rng = np.random.default_rng() if rng is None else rng
    X = []

    while len(X) < N_experiments:
        Y_partial = [
            rng.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]
        Temp = rng.uniform(*TEMP_BOUNDS)
        x = np.asarray(Y_partial + [Temp], dtype=float)

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            X.append(x)

    return np.asarray(X, dtype=float)


def random_design_study(noise_level, X_design):
    """
    Run a fixed random design and estimate parameters cumulatively.

    Parameter estimates start after MIN_ESTIMATION_EXPERIMENTS so the x-axis
    matches BO-style convergence plots.
    """
    print("\n=== RANDOM DESIGN STUDY ===")
    print(f"Noise level: {noise_level:.0e}")

    y_list = []
    Y_tensor_list = []
    param_history = []
    param_exp_counts = []
    current_guess = INITIAL_GUESS.copy()

    for i, x in enumerate(X_design):
        exp_number = i + 1
        print(f"\nExperiment {exp_number}/{len(X_design)}")

        y_scalar, Yexp = run_experiment(x, noise_level=noise_level)
        y_list.append(y_scalar)
        Y_tensor_list.append(Yexp)

        print(f"Output = {y_scalar:.6f}")

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
                noise_level=noise_level,
                initial_guess=current_guess,
            )

            current_guess = params
            param_history.append(params)
            param_exp_counts.append(exp_number)

            print(f"Estimated parameters: {params}")

    y = np.asarray(y_list, dtype=float)
    param_history = np.asarray(param_history, dtype=float)
    param_exp_counts = np.asarray(param_exp_counts, dtype=int)
    Y_full_final = validate_yexp_shape(
        np.concatenate(Y_tensor_list, axis=2),
        nexps_expected=len(X_design),
        name="Y_full_final",
    )

    return {
        "noise": noise_level,
        "X": np.asarray(X_design, dtype=float),
        "y": y,
        "Y_full": Y_full_final,
        "params": param_history,
        "param_exp_counts": param_exp_counts,
        "final_params": param_history[-1] if len(param_history) else None,
    }


# ============================================================
# SUMMARY
# ============================================================

def noise_label(noise):
    return f"{noise:.0e}"


def summarize_result(result):
    X = result["X"]
    y = result["y"]
    final_params = result["final_params"]

    print("\n===== FINAL RANDOM-DESIGN SUMMARY =====")
    print(f"Noise level: {result['noise']:.0e}")
    print(f"Total experiments used: {len(X)}")

    best_idx = int(np.argmax(y))
    print("\nBest experiment:")
    print(f"  X = {X[best_idx]}")
    print(f"  Output = {y[best_idx]:.6f}")

    print(f"\nMean Output = {np.mean(y):.6f}")
    print(f"Max Output  = {np.max(y):.6f}")

    if final_params is not None:
        err = final_params - TRUE_K
        rel_err = np.linalg.norm(err) / np.linalg.norm(TRUE_K)
        print("\nFinal estimated parameters:")
        print(f"  k1 = {final_params[0]:.6f}  error = {err[0]:+.6e}")
        print(f"  k2 = {final_params[1]:.6f}  error = {err[1]:+.6e}")
        print(f"  relative error norm = {rel_err:.6e}")


def print_noise_sweep_table(results):
    print("\n===== RANDOM DESIGN NOISE SWEEP SUMMARY =====\n")
    print("noise      n_exp    k1_final       k2_final       relative_error")
    print("----------------------------------------------------------------")

    for result in results:
        params = result["final_params"]

        if params is None:
            continue

        rel_err = np.linalg.norm(params - TRUE_K) / np.linalg.norm(TRUE_K)
        print(
            f"{result['noise']:.0e}   "
            f"{len(result['X']):5d}   "
            f"{params[0]:12.6f}   "
            f"{params[1]:12.6f}   "
            f"{rel_err:14.6e}"
        )


# ============================================================
# PLOTS
# ============================================================

def apply_plain_axis(ax):
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_random_design(X):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[:, 0],
        X[:, -1],
        c=np.arange(1, len(X) + 1),
        s=100,
    )

    plt.colorbar(label="Experiment number")
    plt.xlabel("Y1")
    plt.ylabel("Temperature (K)")
    plt.title("Shared Random Experimental Design")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_output_all(results):
    plt.figure(figsize=(8, 5))

    for result in results:
        y = result["y"]
        exp_numbers = np.arange(1, len(y) + 1)
        plt.plot(
            exp_numbers,
            y,
            marker="o",
            linewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
        )

    plt.xlabel("Experiment number")
    plt.ylabel("Output fraction")
    plt.title("Output Across Random Experiments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_parameter_convergence_all(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for result in results:
        params = result["params"]
        exp_counts = result["param_exp_counts"]
        label = f"noise {noise_label(result['noise'])}"

        axes[0].plot(exp_counts, params[:, 0], marker="o", label=label)
        axes[1].plot(exp_counts, params[:, 1], marker="s", label=label)

    for i, ax in enumerate(axes):
        ax.axhline(TRUE_K[i], color="black", linestyle="--", linewidth=1.5)
        ax.axhline(INITIAL_GUESS[i], color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Total experiments used")
        ax.set_ylabel(f"k{i + 1}")
        ax.set_title(f"Random Design Parameter Convergence: k{i + 1}")
        ax.grid(True)
        apply_plain_axis(ax)

    axes[0].legend()
    fig.tight_layout()
    plt.show()


def plot_parameter_error_all(results):
    plt.figure(figsize=(8, 5))

    for result in results:
        params = result["params"]
        exp_counts = result["param_exp_counts"]
        errors = np.linalg.norm(params - TRUE_K, axis=1) / np.linalg.norm(TRUE_K)

        plt.plot(
            exp_counts,
            errors,
            marker="o",
            linewidth=1.6,
            label=f"noise {noise_label(result['noise'])}",
        )

    plt.yscale("log")
    plt.xlabel("Total experiments used")
    plt.ylabel("Relative parameter error")
    plt.title("Random Design Parameter Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all(results):
    if results:
        plot_random_design(results[0]["X"])

    plot_output_all(results)
    plot_parameter_convergence_all(results)
    plot_parameter_error_all(results)



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(BASE_SEED)
    X_design = generate_random_design(N_EXPERIMENTS, rng=rng)

    all_results = []

    for noise in NOISE_LEVELS:
        result = random_design_study(noise, X_design)
        summarize_result(result)
        all_results.append(result)

    print_noise_sweep_table(all_results)
    plot_all(all_results)
    
    
    