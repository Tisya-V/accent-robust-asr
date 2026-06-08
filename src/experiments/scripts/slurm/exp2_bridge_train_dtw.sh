#!/bin/bash
#SBATCH --job-name=exp2_bridge_dtw
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

set -e

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

RUNTIME_LOG="logs/%x_%j.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG") 2>&1

echo "=========================================="
echo "E2 Bridge — DTW alignment"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Track: tail -f ${RUNTIME_LOG}"
echo "=========================================="

OUT_DIR=models/bridge_dtw_fixed_x0_0.5

python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train_v2.json \
    --mapping_val_path   src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev_v2.json \
    --alignment       dtw_fixed \
    --cond_acc            \
    --parameterization x0 \
    --out_dir         "${OUT_DIR}" \
    --d_model         768 \
    --dim_feedforward 3072 \
    --n_layers        4 \
    --n_heads         12 \
    --n_epochs        25 \
    --batch_size      64 \
    --lr              1e-4 \
    --weight_decay    1e-4 \
    --sigma_max       0.5 \
    --num_workers     6 \
    --patience        5 \
    --lambda_v        0.0 \
    --notes           "v2 data, i2sb style target formulation and training, now with PEs"

echo "Training done at $(date)"

echo "=========================================="
echo "Running evaluation on test set..."
echo "=========================================="

python -m src.experiments.exp2_latent_diffusion_bridge.eval \
    --mapping_path   src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json \
    --bridge_ckpt    "${OUT_DIR}/checkpoint_best.pt" \
    --predictor_ckpt models/tnat_predictor/model_best.pt \
    --output_dir     results/bridge_eval \
    --output_file    "$(basename "${OUT_DIR}").csv" \
    --n_steps        100 \
    --tnat_buffer    35

echo "Done at $(date)"