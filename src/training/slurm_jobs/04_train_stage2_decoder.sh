#!/bin/bash
#SBATCH --job-name=whisfusion_stage2_decoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem=24GB
#SBATCH --time=04:30:00
#SBATCH --output=logs/whisfusion_stage2_decoder_%j.out
#SBATCH --error=logs/whisfusion_stage2_decoder_%j.err

# Starts Stage 2 training - fine-tunes full decoder and adapter with high masking ratios
# to specialize in initial token generation
#
# Usage on HPC cluster with SLURM:
# 1. chmod +x src/training/slurm_jobs/04_train_stage2_decoder.sh
# 2. sbatch src/training/slurm_jobs/04_train_stage2_decoder.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/whisfusion_stage2_decoder_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Whisfusion Stage 2 Training Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
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
echo "Starting Stage 2: Decoder Specialization (High Mask Ratio, With Perturbations)..."

fabric run src/training/train_stage2_decoder_perturbs.py \
   --strategy=ddp \
   --devices=2 \
   --train_data_dir "${PROCESSED_DATA_DIR}/train/" \
   --val_data_dir   "${PROCESSED_DATA_DIR}/dev/" \
   --pretrain_path  "${MODELS_DIR}/whisfusion_finetuned/stage1_adapter/stage1_adapter.pt" \
   --base_model_path "${MODELS_DIR}/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors" \
   --out_dir        "${MODELS_DIR}/whisfusion_finetuned/stage2_decoder" \
   --model_name     Diff_LLaMA_170M \
   --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
   --num_devices 2              \
   --batch_size 96             \
   --gradient_accumulation_steps 3 \
   --learning_rate 1e-5         \
   --second_stage_lr_multiplier 0.5 \
   --lr_scaling linear          \
   --use_layer_wise_lr_decay    \
   --layer_wise_lr_decay_rate 0.9 \
   --weight_decay 0.005         \
   --scheduler_type cosine      \
   --warmup_ratio 0.1           \
   --epochs 40                  \
   --patience 5                 \
   --use_ema \
   --ema_decay 0.995  \
   --compute_wer_cer   \
   --min_mask_ratio 0.7 \
   --max_mask_ratio 1.0
   # --use_phoneme_perturber \
   # --perturb_prob 0.3 \
   # --include_perturb_in_loss \
   # --perturber_k 10 \
   # --perturber_cache_dir src/utils/cache \

echo "✅ Stage 2 training script finished."
