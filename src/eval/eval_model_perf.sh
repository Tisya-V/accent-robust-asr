#!/bin/bash
#SBATCH --job-name=eval_model_perf
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

python -u -m src.eval.eval_model_perf "$@"

echo "Training completed."