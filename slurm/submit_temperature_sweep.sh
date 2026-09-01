#!/bin/bash
# Run on the login node (not itself a SLURM job): bash slurm/submit_temperature_sweep.sh
# Submits one sbatch job per temperature, each chained to the previous one with
# --dependency=afterany so it only starts once the prior job has finished OR failed.
set -euo pipefail

MODEL_NAME="YOUR_MODEL_NAME"
TEMPERATURES=(0.0 0.3 0.7 1.0 1.3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/run_evaluation.sbatch"
mkdir -p logs

prev_job_id=""
for temp in "${TEMPERATURES[@]}"; do
    dependency_args=()
    if [[ -n "$prev_job_id" ]]; then
        dependency_args=(--dependency=afterany:"$prev_job_id")
    fi

    job_id=$(sbatch --parsable \
        --job-name="misogyny-eval-t${temp}" \
        --export=ALL,MODEL_NAME="$MODEL_NAME",TEMPERATURE="$temp" \
        "${dependency_args[@]}" \
        "$JOB_SCRIPT")

    if [[ -n "$prev_job_id" ]]; then
        echo "Submitted temperature=$temp as job $job_id (runs after job $prev_job_id finishes or fails)"
    else
        echo "Submitted temperature=$temp as job $job_id (runs first)"
    fi

    prev_job_id="$job_id"
done
