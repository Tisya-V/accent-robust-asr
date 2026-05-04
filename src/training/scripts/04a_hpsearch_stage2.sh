#!/bin/bash
#PBS -N hpsearch_stage2_experiment1
#PBS -l select=1:ngpus=3:ncpus=12:mem=96gb
#PBS -l walltime=36:00:00
#PBS -o logs/hpsearch_stage2_experiment1.out
#PBS -e logs/hpsearch_stage2_experiment1.err
#PBS -j oe

# Hyperparameter tuning for Stage 2 decoder specialization
#
# Usage on RDS HPC:
# 1. chmod +x src/training/scripts/04a_hpsearch_stage2.sh
# 2. qsub src/training/scripts/04a_hpsearch_stage2.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

echo "Checking GPU setup..."
nvidia-smi
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_O_WORKDIR=$PBS_O_WORKDIR"
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
echo "Starting Stage 2 HP search..."

python -u -m src.training.src.training.hptuning_ts2_with_perturbs  \
    --resume_existing \
    --hpsearch_dir "${HPSEARCH_DIR}/stage2_decoder_mask_perturb" \
    --trainer_script src/training/src/training/train_stage2_decoder_high_ratio_with_perturbs.py \
    --train_data_dir "${PROCESSED_DATA_DIR}/train/" \
    --val_data_dir "${PROCESSED_DATA_DIR}/dev/" \
    --pretrain_path "${MODELS_DIR}/whisfusion_ft/stage1_adapter/stage1_adapter.pt" \
    --base_model_path "${MODELS_DIR}/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors" \
    --out_model_name whisfusion \
    --model_name Diff_LLaMA_170M \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --perturber_cache_dir src/utils/cache \
    --num_devices 3 \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --epochs 40 \
    --learning_rate 1e-5 \
    --second_stage_lr_multiplier 0.5 \
    --lr_scaling linear \
    --weight_decay 0.005 \
    --scheduler_type cosine \
    --warmup_ratio 0.1 \
    --patience 5 \
    --use_ema \
    --ema_decay 0.995 \
    --compute_wer_cer \
    --use_layer_wise_lr_decay \
    --layer_wise_lr_decay_rate 0.9 \
    --gradient_clip_val 1.0 \
    --precision 32-true \
    --val_steps 0 \
    --num_workers 8 \
    --early_stop_metric loss

echo "✅ Stage 2 HP search script finished."
