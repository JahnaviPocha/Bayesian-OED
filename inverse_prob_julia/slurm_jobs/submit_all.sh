#!/bin/bash
# Submits every bayes_design*.job in this directory so all runs launch concurrently.
cd "$(dirname "$0")" || exit 1
for job in bayes_design*.job; do
    sbatch "$job"
done
