#!/bin/bash
#SBATCH --job-name=whisper_ctc_aux
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH

source activate

source /vol/cuda/12.0.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

nvidia-smi

python -u -m src.training.train_aux \
    --output_dir models/ctc_aux
    # --epochs 5 \
    # --batch_size 8 \
    # --lr 1e-4 \
    # --lora_r 8

echo "Training completed."