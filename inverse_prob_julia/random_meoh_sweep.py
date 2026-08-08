#!/usr/bin/env python3
"""
Random-design methanol sweep, one independent task per (Nexps, noise).

complete_workflow in scripts/main_meoh.jl is a whole study on its own: it
builds the Halton design for Nexps points, generates the data at std_data, and
estimates the parameters from a fixed initial guess of Par * 0.5. Only Nexps
and std_data vary between runs, and nothing carries over from one run to the
next, so the 141 experiment counts x 3 noise levels are 423 independent
problems rather than a sequence.

The sequential script walks them in order and warm-starts each fit from the
previous one, which serialises the whole sweep for no reason other than that
warm start. Here every task stands alone.

    python random_meoh_sweep.py --stage sweep --rbs-full --shard 0 --nshards 12
    python random_meoh_sweep.py --stage collect --rbs-full

Each task checkpoints on completion, so a shard that dies costs only its
in-flight task and a resubmission skips whatever is already done. The collect
stage needs no Julia and runs on the login node.

Two properties of complete_workflow make this work:

  Halton is nested, so the design for Nexps=10 is the first 10 points of the
  design for Nexps=150. Experiment counts are directly comparable.

  The initial guess is Par * 0.5 regardless of Nexps, so no task depends on
  another's result.

One consequence worth being explicit about: each task calls experiments()
itself, so every experiment count gets its own noise draw. The sequential
script instead grows one dataset, where the point at n=20 contains the same
measurements as the point at n=10 plus ten more. Both are legitimate, but they
answer slightly different questions. This one gives independent replicates at
each count, so the curve is noisier point to point and free of the correlation
that makes a nested sweep look smoother than its information content warrants.

Tasks within a shard run in ascending Nexps order on purpose. The offline
cache in main_meoh.jl is keyed on the design point, and Halton nesting means a
task at Nexps=150 reuses everything computed for the smaller counts in the same
process, so each shard pays for its offline solves once.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

NOISE_LEVELS = [1e-5, 1e-6, 1e-7]
MIN_EXPERIMENTS = 10
MAX_EXPERIMENTS = 150
SCALE = 1.0
RATIO = 0.1
N_PARAMETERS = 9

TRUE_PARAMS = np.array(
    [15672.02, 3453.38, 30.836, 558.532, 0.7439, 40000, 17197, 124119, 98084],
    dtype=float,
)
PARAM_LOWER = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1e4, 1e4, 1e4, 1e4], dtype=float)
PARAM_UPPER = np.array(
    [1e5, 1e4, 1e5, 1e5, 1e5, 1.5e5, 1.5e5, 1.5e5, 1.5e5], dtype=float
)
PARAMETER_SHORT_NAMES = [
    "A1", "E1", "K_ads,1", "K_ads,2", "A2", "E2", "K_ads,3", "K_ads,4",
    "Inhibition / adsorption",
]


def results_dir(rbs_full):
    name = "random_meoh_sweep_RBS" if rbs_full else "random_meoh_sweep_asymptotic"
    return PROJECT_DIR / "results" / name


def normalize(params):
    params = np.asarray(params, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log10(params) - np.log10(PARAM_LOWER)) / (
            np.log10(PARAM_UPPER) - np.log10(PARAM_LOWER)
        )


def task_list(shard, nshards):
    """
    Deal tasks largest-first so the expensive counts start immediately, then
    run each shard in ascending order so its offline cache builds up instead of
    being warmed by the largest task alone.
    """
    tasks = [
        (noise, n)
        for noise in NOISE_LEVELS
        for n in range(MIN_EXPERIMENTS, MAX_EXPERIMENTS + 1)
    ]
    tasks.sort(key=lambda t: -t[1])
    mine = tasks[shard::nshards]
    mine.sort(key=lambda t: t[1])
    return mine


# ============================================================
# SWEEP
# ============================================================

def stage_sweep(rbs_full, shard, nshards):
    from call_to_KPE_code_meoh import complete_workflow

    out = results_dir(rbs_full) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)

    tasks = task_list(shard, nshards)
    print(f"shard {shard}/{nshards}: {len(tasks)} tasks, "
          f"Nexps {tasks[0][1]}..{tasks[-1][1]}, RBS_full={rbs_full}", flush=True)

    failed = []
    for index, (noise, n) in enumerate(tasks, start=1):
        path = out / f"n{n:03d}_noise{noise:.0e}.npz".replace("-", "m")
        if path.exists():
            print(f"[{index}/{len(tasks)}] n={n} noise={noise:.0e}: cached", flush=True)
            continue

        started = time.time()
        try:
            params = complete_workflow(
                scale=SCALE,
                Nexps=n,
                ratio=RATIO,
                nparas=N_PARAMETERS,
                std_data=noise,
                RBS_full=rbs_full,
            )
            params = np.asarray(params, dtype=float).reshape(-1)
        except Exception as exc:                      # keep the shard alive
            failed.append((noise, n, repr(exc)))
            print(f"[{index}/{len(tasks)}] n={n} noise={noise:.0e}: FAILED {exc!r}",
                  flush=True)
            continue

        elapsed = time.time() - started
        rms = float(
            np.linalg.norm(normalize(params) - normalize(TRUE_PARAMS))
            / np.sqrt(N_PARAMETERS)
        )

        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, params=params, n_experiments=n, noise=noise,
                            seconds=elapsed, rms_normalized_error=rms)
        tmp.replace(path)

        print(f"[{index}/{len(tasks)}] n={n} noise={noise:.0e}: "
              f"{elapsed / 60:.1f} min, rms={rms:.4f}", flush=True)

    if failed:
        print(f"\n{len(failed)} tasks failed in this shard:", flush=True)
        for noise, n, error in failed:
            print(f"  n={n} noise={noise:.0e}: {error}", flush=True)


# ============================================================
# COLLECT
# ============================================================

def stage_collect(rbs_full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    base = results_dir(rbs_full)
    files = sorted((base / "checkpoints").glob("n*.npz"))
    if not files:
        sys.exit(f"no checkpoints in {base / 'checkpoints'}")

    rows = []
    for path in files:
        with np.load(path) as data:
            params = data["params"]
            noise = float(data["noise"])
            n = int(data["n_experiments"])
            for j in range(N_PARAMETERS):
                rows.append({
                    "noise": f"{noise:.0e}",
                    "n_experiments": n,
                    "parameter": f"p{j + 1}",
                    "name": PARAMETER_SHORT_NAMES[j],
                    "estimate": params[j],
                    "true_value": TRUE_PARAMS[j],
                    "relative_error": (params[j] - TRUE_PARAMS[j]) / TRUE_PARAMS[j],
                    "rms_normalized_error": float(data["rms_normalized_error"]),
                    "seconds": float(data["seconds"]),
                })

    frame = pd.DataFrame(rows)
    base.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base / "parameter_history.csv", index=False)
    with pd.ExcelWriter(base / "variable_explorer_data.xlsx") as writer:
        frame.to_excel(writer, sheet_name="parameter_history", index=False)

    noises = sorted(frame["noise"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for noise in noises:
        sub = frame[frame["noise"] == noise].drop_duplicates("n_experiments")
        sub = sub.sort_values("n_experiments")
        ax.plot(sub["n_experiments"], sub["rms_normalized_error"],
                marker="o", markersize=3, linewidth=1.4, label=f"noise {noise}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel("RMS normalized log-parameter error")
    ax.set_title(f"Random design sweep ({'RBS' if rbs_full else 'asymptotic'})")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(base / "01_rms_parameter_error.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rows_n = int(np.ceil(N_PARAMETERS / 3))
    fig, axes = plt.subplots(rows_n, 3, figsize=(15, 4 * rows_n), squeeze=False)
    for j in range(N_PARAMETERS):
        ax = axes[j // 3][j % 3]
        for noise in noises:
            sub = frame[(frame["noise"] == noise) & (frame["parameter"] == f"p{j+1}")]
            sub = sub.sort_values("n_experiments")
            ax.plot(sub["n_experiments"], sub["estimate"],
                    marker="o", markersize=3, linewidth=1.2, label=f"noise {noise}")
        ax.axhline(TRUE_PARAMS[j], color="black", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"p{j+1}: {PARAMETER_SHORT_NAMES[j]}")
        ax.set_xlabel("Number of experiments")
        ax.grid(True, alpha=0.4)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(N_PARAMETERS, rows_n * 3):
        axes[j // 3][j % 3].axis("off")
    fig.suptitle("Parameter convergence, dashed line is the true value")
    fig.tight_layout()
    fig.savefig(base / "02_parameter_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    done = frame.drop_duplicates(["noise", "n_experiments"])
    print(f"tasks collected : {len(done)} of "
          f"{len(NOISE_LEVELS) * (MAX_EXPERIMENTS - MIN_EXPERIMENTS + 1)}")
    for noise in noises:
        sub = done[done["noise"] == noise]
        print(f"  noise {noise}: {len(sub)} counts, "
              f"n up to {sub['n_experiments'].max()}, "
              f"best rms {sub['rms_normalized_error'].min():.4f}")
    print(f"total compute   : {done['seconds'].sum() / 3600:.1f} h")
    print(f"\nwritten to {base}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["sweep", "collect"])
    parser.add_argument("--rbs-full", action="store_true",
                        help="RBS reduced basis; omit for the asymptotic model")
    parser.add_argument("--shard", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--nshards", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()

    if args.stage == "sweep":
        stage_sweep(args.rbs_full, args.shard, args.nshards)
    else:
        stage_collect(args.rbs_full)


if __name__ == "__main__":
    main()
