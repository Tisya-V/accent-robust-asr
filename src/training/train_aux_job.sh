#!/bin/bash
#SBATCH --job-name=whisper_feat_aux0p2
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH

source activate

source /vol/cuda/12.4.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

nvidia-smi

python -u -m src.training.train_aux \
    --output_dir models/feat_aux0p2 \
    --ctc_layer 7 \
    --lambda_ctc 0.00 \
    --lambda_feat 0.20 \
    --epochs 10 \
    --feat_layer 11 
    # --held_out_l1 Chinese \

echo "Training completed."