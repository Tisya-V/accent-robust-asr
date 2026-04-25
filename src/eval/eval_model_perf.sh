#!/bin/bash
#SBATCH --job-name=eval_model_perf_baselines_spontaneous
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euxo pipefail 

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH

source activate

source /vol/cuda/12.4.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

nvidia-smi

echo "=== Job started ===" | tee -a ${SLURM_SUBMIT_DIR}/logs/start_${SLURM_JOB_ID}.log
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "hostname: $(hostname)"
nvidia-smi
echo "python --version: $(python --version)"
which python
echo "PATH first 3: ${PATH%%:*} ${PATH#*:} ${PATH%%:*${PATH#*:*}}"
echo "====================="

srun python -u -m src.eval.eval_model_perf \
    --models "baseline, no_aux, no_aux_heldout_chinese" \
    --split spontaneous

echo "Evaluation completed."