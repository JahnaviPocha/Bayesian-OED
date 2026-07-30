#!/usr/bin/env python3
"""
Random-design methanol study, restructured so the estimations run in parallel.

The sequential version walks experiments 1..100 in order and re-fits the
parameters after each one. Those fits are independent problems: the fit at n
uses the first n experiments and nothing from the fit at n-1 except a warm
start. Running them in lockstep therefore serialises ~800 h of work per noise
level for no reason, and the per-fit cost grows with n (the Julia objective
loops over all Nexps), so the tail dominates.

This script splits the work into three stages:

  experiments  run the 100 forward simulations for each noise level and store
               the measurement tensors. Independent across (noise, experiment).
  estimate     fit parameters for every (noise, n) with n = 10..100, in
               parallel. Independent across tasks; shardable over a job array.
  collect      assemble the checkpoints, then write the Excel file and the
               plots using the original script's own functions.

Every task checkpoints to results/<folder>/checkpoints/ the moment it finishes,
so a wall-clock kill costs at most the tasks still in flight and a resubmission
picks up where it stopped.

    python bayes_design_random_meoh_RBS_parallel.py --stage experiments
    python bayes_design_random_meoh_RBS_parallel.py --stage estimate --shard 0 --nshards 12
    python bayes_design_random_meoh_RBS_parallel.py --stage collect

One deliberate numerical change: the sequential version seeds each fit with the
previous fit's result, which the parallel form cannot do. Every fit here starts
from INITIAL_GUESS_NORMALIZED, the same guess the sequential run uses for its
first fit. Estimates at a given n may therefore differ from the sequential run,
and each fit does more Newton iterations. The trade is deliberate: the chain is
what forced the work to be serial.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import bayes_design_random_meoh_RBS as base

RESULTS_FOLDER = "bayes_design_random_meoh_RBS_parallel"
WORK_DIR = base.PROJECT_DIR / "results" / RESULTS_FOLDER
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
DESIGN_PATH = WORK_DIR / "design.npz"


def noise_tag(noise):
    return f"{noise:.0e}".replace("-", "m").replace("+", "p")


def default_workers():
    """
    One worker per JULIA_NUM_THREADS cores of the allocation.

    Julia threads over Nexps inside the objective, so each worker wants its
    own small thread pool rather than one worker grabbing every core.
    """
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    threads = int(os.environ.get("JULIA_NUM_THREADS", 4))
    return max(1, cpus // max(1, threads))


# ============================================================
# STAGE 1: FORWARD EXPERIMENTS
# ============================================================

def experiment_task(noise, index, x):
    path = CHECKPOINT_DIR / f"exp_{noise_tag(noise)}_{index:03d}.npz"
    if path.exists():
        return noise, index, "cached"

    started = time.time()
    methanol, Yexp = base.run_experiment(x, noise)
    np.savez_compressed(path, methanol=methanol, Yexp=Yexp)
    return noise, index, f"{time.time() - started:.1f}s"


def stage_experiments(workers):
    rng = np.random.default_rng(base.BASE_SEED)
    X_design = base.generate_random_design(base.N_EXPERIMENTS, rng)
    np.savez_compressed(DESIGN_PATH, X_design=X_design)
    print(f"design: {X_design.shape[0]} points, saved to {DESIGN_PATH}", flush=True)

    tasks = [
        (noise, index, X_design[index])
        for noise in base.NOISE_LEVELS
        for index in range(base.N_EXPERIMENTS)
    ]
    print(f"running {len(tasks)} forward simulations on {workers} workers", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(experiment_task, *task) for task in tasks]
        for future in as_completed(futures):
            noise, index, note = future.result()
            done += 1
            print(f"[{done}/{len(tasks)}] noise {noise:.0e} experiment {index + 1}: {note}",
                  flush=True)

    for noise in base.NOISE_LEVELS:
        collect_experiment_tensor(noise, X_design.shape[0], save=True)


def collect_experiment_tensor(noise, n_experiments, save=False):
    """Stack the per-experiment checkpoints into the (repeats, spec, n) tensor."""
    y = np.full(n_experiments, np.nan)
    tensors = []

    for index in range(n_experiments):
        path = CHECKPOINT_DIR / f"exp_{noise_tag(noise)}_{index:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing forward experiment: {path}")
        with np.load(path) as data:
            y[index] = float(data["methanol"])
            tensors.append(data["Yexp"])

    Y_full = np.concatenate(tensors, axis=2)
    if save:
        out = WORK_DIR / f"experiments_{noise_tag(noise)}.npz"
        np.savez_compressed(out, y=y, Y_full=Y_full)
        print(f"noise {noise:.0e}: tensor {Y_full.shape} -> {out}", flush=True)
    return y, Y_full


# ============================================================
# STAGE 2: PARAMETER ESTIMATION
# ============================================================

def estimate_task(noise, n_used):
    """
    Fit the parameters using the first n_used experiments.

    Loads the tensor inside the worker rather than receiving it as an argument,
    so the array is not pickled once per task.
    """
    path = CHECKPOINT_DIR / f"est_{noise_tag(noise)}_{n_used:03d}.npz"
    if path.exists():
        return noise, n_used, "cached"

    with np.load(DESIGN_PATH) as data:
        X_design = data["X_design"]
    with np.load(WORK_DIR / f"experiments_{noise_tag(noise)}.npz") as data:
        Y_full = data["Y_full"]

    started = time.time()
    params = base.estimate_parameters(
        X_design[:n_used],
        Y_full[:, :, :n_used],
        noise,
        initial_guess=base.INITIAL_GUESS_NORMALIZED,
    )
    elapsed = time.time() - started
    np.savez_compressed(path, params=params, n_used=n_used, noise=noise, seconds=elapsed)
    return noise, n_used, f"{elapsed / 60:.1f} min"


def build_task_list(shard, nshards):
    """
    Deal tasks largest-first so the long fits start immediately.

    Cost grows steeply with n, so a shard that received only small n would
    finish early while another still held n=100. Round-robin over the
    descending list balances the shards.
    """
    tasks = [
        (noise, n)
        for noise in base.NOISE_LEVELS
        for n in range(base.MIN_ESTIMATION_EXPERIMENTS, base.N_EXPERIMENTS + 1)
    ]
    tasks.sort(key=lambda t: -t[1])
    return tasks[shard::nshards]


def stage_estimate(workers, shard, nshards):
    tasks = build_task_list(shard, nshards)
    print(f"shard {shard}/{nshards}: {len(tasks)} fits on {workers} workers "
          f"(n from {min(t[1] for t in tasks)} to {max(t[1] for t in tasks)})", flush=True)

    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(estimate_task, *task): task for task in tasks}
        for future in as_completed(futures):
            noise, n_used = futures[future]
            done += 1
            try:
                _, _, note = future.result()
                print(f"[{done}/{len(tasks)}] noise {noise:.0e} n={n_used}: {note}", flush=True)
            except Exception as exc:                     # keep the shard alive
                failed.append((noise, n_used, repr(exc)))
                print(f"[{done}/{len(tasks)}] noise {noise:.0e} n={n_used}: FAILED {exc!r}",
                      flush=True)

    if failed:
        print(f"\n{len(failed)} fits failed in this shard:", flush=True)
        for noise, n_used, error in failed:
            print(f"  noise {noise:.0e} n={n_used}: {error}", flush=True)


# ============================================================
# STAGE 3: COLLECTION AND OUTPUT
# ============================================================

def stage_collect():
    with np.load(DESIGN_PATH) as data:
        X_design = data["X_design"]

    results = []
    for noise in base.NOISE_LEVELS:
        y, _ = collect_experiment_tensor(noise, X_design.shape[0])

        params, counts = [], []
        for n_used in range(base.MIN_ESTIMATION_EXPERIMENTS, base.N_EXPERIMENTS + 1):
            path = CHECKPOINT_DIR / f"est_{noise_tag(noise)}_{n_used:03d}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                params.append(data["params"])
                counts.append(n_used)

        if not params:
            print(f"noise {noise:.0e}: no estimates found, skipping", flush=True)
            continue

        missing = (base.N_EXPERIMENTS - base.MIN_ESTIMATION_EXPERIMENTS + 1) - len(counts)
        if missing:
            print(f"noise {noise:.0e}: {missing} estimates missing, "
                  f"plotting the {len(counts)} available", flush=True)

        results.append({
            "noise": noise,
            "X": X_design,
            "y": y,
            "params": np.asarray(params, dtype=float),
            "param_exp_counts": np.asarray(counts, dtype=int),
            "final_params": np.asarray(params[-1], dtype=float),
        })

    if not results:
        sys.exit("nothing to collect")

    # The plotting and export helpers write to the module-level RESULTS_DIR.
    base.RESULTS_DIR = WORK_DIR

    base.print_summary(results)
    base.export_all_outputs(results, X_design)
    base.plot_design(X_design)
    base.plot_outputs(results)
    base.plot_parameter_physical(results)
    base.plot_parameter_normalized(results)
    base.plot_error(results)
    print(f"\nwritten to {WORK_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=["experiments", "estimate", "collect"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--nshards", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    workers = args.workers or default_workers()

    if args.stage == "experiments":
        stage_experiments(workers)
    elif args.stage == "estimate":
        stage_estimate(workers, args.shard, args.nshards)
    else:
        stage_collect()


if __name__ == "__main__":
    mp.freeze_support()
    main()
