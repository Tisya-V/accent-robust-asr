#!/bin/bash
#SBATCH --job-name=hpsearch_experiment1
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib
ifconfig -a | grep -E 'UP|inet ' | grep -v lo

export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=enoinp0
# export NCCL_DEBUG=WARN

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH
source activate

source /vol/cuda/12.4.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

set -e

echo "Checking GPU/SLURM setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("DEBUG CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
   print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

echo -e "

==============================

"

echo "Starting Stage 2 HP search..."

python -u -m src.training.src.training.hptuning_ts2_with_perturbs  \
    --resume_existing \
    --hpsearch_dir hpsearch/stage2_decoder_mask_perturb \
    --trainer_script src/training/src/training/train_stage2_decoder_high_ratio_with_perturbs.py \
    --train_data_dir data/processed/train/   \
    --val_data_dir data/processed/dev/   \
    --pretrain_path models/whisfusion_ft/stage1_adapter/stage1_adapter.pt   \
    --base_model_path models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors   \
    --out_model_name whisfusion   \
    --model_name Diff_LLaMA_170M   \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T   \
    --perturber_cache_dir src/utils/cache   \
    --num_devices 3   \
    --batch_size 48   \
    --gradient_accumulation_steps 2   \
    --epochs 40   \
    --learning_rate 1e-5   \
    --second_stage_lr_multiplier 0.5   \
    --lr_scaling linear   \
    --weight_decay 0.005   \
    --scheduler_type cosine   \
    --warmup_ratio 0.1   \
    --patience 5   \
    --use_ema   \
    --ema_decay 0.995   \
    --compute_wer_cer   \
    --use_layer_wise_lr_decay   \
    --layer_wise_lr_decay_rate 0.9   \
    --gradient_clip_val 1.0   \
    --precision 32-true   \
    --val_steps 0   \
    --num_workers 8   \
    --early_stop_metric loss

echo "✅ Stage 2 HP search script finished."
