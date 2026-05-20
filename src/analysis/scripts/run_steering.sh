#!/usr/bin/env bash
#SBATCH --job-name=steering_whisper_position
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

echo "[Job] Starting at $(date)"
echo "[Job] Host: $(hostname)"
echo "[Job] GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

mkdir -p logs

python src/analysis/run_steering.py \
    --decoder whisper \
    --method position \
    --out_dir results/e2_steering \
    --num_per_l1 100

echo "[Job] Done at $(date)"
