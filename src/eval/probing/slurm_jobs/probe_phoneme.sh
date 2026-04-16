#!/bin/bash
#SBATCH --job-name=probe_phoneme
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib


export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH
source activate

source /vol/cuda/12.0.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

nvidia-smi

for MODEL in "baseline" "no_aux" "ctc_aux_l3" "ctc_aux_l7" "feat_aux" "feat_aux0p3"; do
    echo -e "\n\nEvaluating model: $MODEL\n"
    python -u -m src.eval.probing.probe_phoneme --models $MODEL 
done

echo "Phoneme probe evaluation completed."