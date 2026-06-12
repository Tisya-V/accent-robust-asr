
#!/bin/bash
#SBATCH --job-name=exp2_bridge_dtw_eps_relog
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

echo "=========================================="
echo "E2 Bridge — DTW-fixed alignment, eps, sigma_max=0.5, seed=42 (isolated relog)"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=========================================="

OUT_DIR=models/bridge_dtw_fixed_eps_0.5_seed42_relog

python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json \
    --mapping_val_path   src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json \
    --alignment       dtw_fixed \
    --cond_acc            \
    --parameterization eps \
    --out_dir         "${OUT_DIR}" \
    --d_model         768 \
    --dim_feedforward 3072 \
    --n_layers        4 \
    --n_heads         12 \
    --n_epochs        35 \
    --batch_size      16 \
    --lr              1e-4 \
    --weight_decay    1e-4 \
    --sigma_max       0.5 \
    --num_workers     6 \
    --patience        8 \
    --ema_decay       0.99 \
    --notes           "isolated norm-tracking relog (eps), do not overwrite existing models" \
    --seed            42

echo "Training done at $(date)"
