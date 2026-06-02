#!/bin/bash
#SBATCH --job-name=exp2_bridge_eval
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp2_bridge_eval_%j.out
#SBATCH --error=logs/exp2_bridge_eval_%j.out

# Evaluate E2 Latent Diffusion Bridge model on SLURM
#
# Usage on SLURM cluster:
# 1. chmod +x src/experiments/scripts/slurm/exp2_bridge_eval.sh
# 2. sbatch src/experiments/scripts/slurm/exp2_bridge_eval.sh

set -e

# Source centralized environment configuration
source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "E2 Latent Diffusion Bridge Evaluation Job"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_GPUS=$SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("DEBUG CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

echo -e "\n\n==============================\n\n"
echo "Starting Bridge Evaluation..."

MODELS=( "bridge_dtw_eps_sde_sig1.5" )

for MODEL in "${MODELS[@]}"; do
    python -m src.experiments.exp2_latent_diffusion_bridge.eval \
        --bridge_ckpt    models/"$MODEL"/checkpoint_best.pt \
        --predictor_ckpt models/tnat_predictor/model_best.pt \
        --output_dir     results/bridge_eval \
        --output_file    "$MODEL".csv \
        --n_steps        100 \
        --max_utts_per_speaker 50 \
        --tnat_buffer   35
done

echo "✅ Bridge evaluation complete."
