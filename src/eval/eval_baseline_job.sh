#!/bin/bash
#SBATCH --job-name=whisper_baseline
#SBATCH --partition=a30         # GPU partition (adjust)
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH --time=02:00:00         # 30min max
#SBATCH --output=logs/%x_%j.out # Output log
#SBATCH --error=logs/%x_%j.out  # Error log


export HF_HOME=/vol/bitbucket/tsv22/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/tsv22/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/tsv22/.cache

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH
source activate

source /vol/cuda/12.0.0/setup.sh

# Run
cd /vol/bitbucket/$USER/accent-robust-asr/
python -m src.eval.eval_baseline

echo "Baseline evaluation completed."