#!/usr/bin/env bash
#SBATCH --job-name=steering_whisper
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

echo "[Job] Starting at $(date)"
echo "[Job] Host: $(hostname)"
echo "[Job] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

mkdir -p logs

# whisper_position already cached — skipped automatically by run_steering.py

# for TAIL in l2 english interpolate; do
#     echo "[Job] whisper dtw tail=${TAIL}"
#     python src/analysis/run_steering.py \
#         --decoder whisper --method dtw --tail "${TAIL}" \
#         --num_prompts 100 --out_dir results/e2_steering
# done


echo "[Job] whisper position_fixed"
python src/analysis/run_steering.py \
    --decoder whisper --method dtw_fixed \
    --num_prompts 100 --out_dir results/e2_steering

echo "[Job] Done at $(date)"
