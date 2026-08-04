#!/usr/bin/env python3
"""
Build plots and tables from a running job's progress files.

The BO scripts now write results/<run>/progress/noise_*.npz after every BO
iteration. This turns whatever is there right now into the same CSV, Excel and
plots the finished run would produce, without importing juliacall, so it can be
run on the login node at any point while the job is still going.

    python tools/snapshot_progress.py results/bayes_design_meoh_RBS_20260804_190000

Output goes to <run>/snapshot/. Re-run it whenever you want a fresher view; it
overwrites its own snapshot folder and touches nothing else.

Unlike tools/extract_partial_results.py, which scrapes the SLURM log and has to
guess which noise level a line belongs to, each progress file holds one noise
level's arrays directly. Nothing is inferred, and for the FIO run the chosen
designs are present here even though they are never printed to the log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_partial_results import (          # noqa: E402
    PROJECT_DIR,
    load_script_constants,
    normalize_parameters,
    plot_all,
)


def script_for_run(run_dir):
    """
    Map a results folder back to the script that produced it.

    Folders are named after the script plus a timestamp, so the longest
    matching script name wins: bayes_design_meoh_FIO_RBS must be preferred
    over bayes_design_meoh_RBS, which is a prefix of nothing but could be
    matched loosely.
    """
    name = Path(run_dir).name
    candidates = sorted(PROJECT_DIR.glob("bayes_design_*.py"),
                        key=lambda p: -len(p.stem))
    for script in candidates:
        if name.startswith(script.stem):
            return script
    sys.exit(f"cannot tell which script produced {name}")


def build_frames(progress_files, constants):
    n_params = int(constants.get("N_UNKNOWN_PARAMETERS", 9))
    names = constants.get("PARAMETER_SHORT_NAMES", [f"p{i+1}" for i in range(n_params)])
    true_params = np.asarray(constants["TRUE_PARAMS"], dtype=float)
    lower = np.asarray(constants["PARAM_LOWER"], dtype=float)
    upper = np.asarray(constants["PARAM_UPPER"], dtype=float)
    true_norm = normalize_parameters(true_params, lower, upper)

    history_rows, experiment_rows = [], []

    for path in sorted(progress_files):
        with np.load(path) as data:
            noise = float(data["noise"])
            label = f"{noise:.0e}"
            X = data["X"]
            y = data["y"]
            params = data["params"]
            counts = data["param_exp_counts"]
            info = data["total_info_history"] if "total_info_history" in data else None
            point_info = data["point_information"] if "point_information" in data else None

            for i in range(len(y)):
                row = {"noise": label, "experiment": i + 1, "methanol": float(y[i])}
                if point_info is not None and i < len(point_info):
                    row["point_information"] = float(point_info[i])
                for j, value in enumerate(X[i]):
                    row[f"x{j + 1}"] = float(value)
                experiment_rows.append(row)

            for i, n_exp in enumerate(counts):
                estimate = np.asarray(params[i], dtype=float)
                rms = float(
                    np.linalg.norm(normalize_parameters(estimate, lower, upper) - true_norm)
                    / np.sqrt(n_params)
                )
                for j in range(n_params):
                    history_rows.append({
                        "noise": label,
                        "total_experiments_used": int(n_exp),
                        "parameter": f"p{j + 1}",
                        "name": names[j],
                        "estimate": estimate[j],
                        "true_value": true_params[j],
                        "relative_error": (estimate[j] - true_params[j])
                                          / max(abs(true_params[j]), 1e-30),
                        "rms_normalized_parameter_error": rms,
                        "cumulative_information": (float(info[i])
                                                   if info is not None and i < len(info)
                                                   else None),
                    })

    return pd.DataFrame(history_rows), pd.DataFrame(experiment_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="results/<run> folder")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    progress_dir = args.run_dir / "progress"
    files = sorted(progress_dir.glob("noise_*.npz"))
    if not files:
        sys.exit(f"no progress files in {progress_dir}")

    constants = load_script_constants(script_for_run(args.run_dir))
    history, experiments = build_frames(files, constants)

    out_dir = args.out or (args.run_dir / "snapshot")
    out_dir.mkdir(parents=True, exist_ok=True)

    history.to_csv(out_dir / "parameter_history.csv", index=False)
    experiments.to_csv(out_dir / "experiments.csv", index=False)
    with pd.ExcelWriter(out_dir / "variable_explorer_data.xlsx") as writer:
        history.to_excel(writer, sheet_name="parameter_history", index=False)
        experiments.to_excel(writer, sheet_name="experiments", index=False)

    if not args.no_plots and not history.empty:
        plot_all(history, experiments, constants, out_dir)

    print(f"run       : {args.run_dir.name}")
    print(f"noise     : {', '.join(sorted(history['noise'].unique()))}")
    for noise in sorted(history["noise"].unique()):
        subset = history[history["noise"] == noise]
        print(f"  {noise}: {subset['total_experiments_used'].max()} experiments, "
              f"{subset['total_experiments_used'].nunique()} estimations")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
