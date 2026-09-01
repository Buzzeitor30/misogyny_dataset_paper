#!/bin/bash
#SBATCH --job-name=misogyny-eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

: "${MODEL_NAME:?MODEL_NAME must be set (passed via sbatch --export)}"
: "${TEMPERATURE:?TEMPERATURE must be set (passed via sbatch --export)}"

echo "Job $SLURM_JOB_ID: evaluating $MODEL_NAME at temperature=$TEMPERATURE"

uv run python evaluate.py \
    --model_name "$MODEL_NAME" \
    --temperature "$TEMPERATURE"
