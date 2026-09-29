#!/bin/bash
#SBATCH --job-name=misogyny-finetune
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ~/.hf_token

# Defaults run the first target: task1 (misogyny presence), r=16 (alpha=2r), lr=1e-4.
# Override through sbatch --export, e.g.:
#   sbatch --export=ALL,TASK=all slurm/run_finetune.sbatch
TASK="${TASK:-task1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
LORA_RANK="${LORA_RANK:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

echo "Job $SLURM_JOB_ID: finetuning $MODEL_NAME on $TASK (rank=$LORA_RANK, lr=$LEARNING_RATE)"

# --output/--error above point into logs/; create it so a direct sbatch of this file works.
mkdir -p logs

uv run python scripts/finetune_llm.py \
    --task "$TASK" \
    --model_name "$MODEL_NAME" \
    --lora_rank "$LORA_RANK" \
    --learning_rate "$LEARNING_RATE"
