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

MODELS=( "bridge_dtw_fixed_x0_1.0" "bridge_dtw_fixed_eps_1.0" )
# REVERT THE CHECKPOINTS TO THE BEST ONES BEFORE RUNNING THIS SCRIPT
for MODEL in "${MODELS[@]}"; do
    # echo "SDE inference"
    # python -m src.experiments.exp2_latent_diffusion_bridge.eval \
    #     --mapping_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json \
    #     --bridge_ckpt    models/"$MODEL"/checkpoint_latest.pt \
    #     --predictor_ckpt models/tnat_predictor/model_best.pt \
    #     --output_dir     results/bridge_eval \
    #     --output_file    "$MODEL".csv \
    #     --n_steps        100 \
    #     --tnat_buffer   35
    #     # --max_utts_per_speaker 50 \

    echo "SDE inference with renorm"
    python -m src.experiments.exp2_latent_diffusion_bridge.eval \
        --mapping_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json \
        --bridge_ckpt    models/"$MODEL"/checkpoint_latest.pt \
        --predictor_ckpt models/tnat_predictor/model_best.pt \
        --output_dir     results/bridge_eval \
        --output_file    "$MODEL"_renorm.csv \
        --n_steps        100 \
        --tnat_buffer   35 \
        --norm_renorm   \
        # --max_utts_per_speaker 50 \

    # echo "ODE inference without renormalization"
    # python -m src.experiments.exp2_latent_diffusion_bridge.eval \
    #     --mapping_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json \
    #     --bridge_ckpt    models/"$MODEL"/checkpoint_latest.pt \
    #     --predictor_ckpt models/tnat_predictor/model_best.pt \
    #     --output_dir     results/bridge_eval \
    #     --output_file    "$MODEL"_ode.csv \
    #     --n_steps        100 \
    #     --tnat_buffer   35 \
    #     --ode_sampling

    # echo "ODE inference with renormalization"
    # python -m src.experiments.exp2_latent_diffusion_bridge.eval \
    #     --mapping_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json \
    #     --bridge_ckpt    models/"$MODEL"/checkpoint_latest.pt \
    #     --predictor_ckpt models/tnat_predictor/model_best.pt \
    #     --output_dir     results/bridge_eval \
    #     --output_file    "$MODEL"_ode_renorm.csv \
    #     --n_steps        100 \
    #     --tnat_buffer   35 \
    #     --ode_sampling   \
    #     --norm_renorm   \
    #     # --max_utts_per_speaker 50 \
done

echo "✅ Bridge evaluation complete."
