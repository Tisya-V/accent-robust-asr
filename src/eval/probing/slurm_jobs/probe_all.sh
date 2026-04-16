#!/bin/bash
#SBATCH --job-name=probe_all_no_aux
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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
MODEL="no_aux"

echo -e "\n\nEvaluating model: $MODEL\n"

echo "Running clustering analysis"
python -u -m src.eval.probing.probe_clustering --models $MODEL
echo "Clustering evaluation completed."

echo "Running accent probe"
python -u -m src.eval.probing.probe_accent --models $MODEL --within_phoneme
echo "Accent probe evaluation completed."

echo "Running phoneme probe"
python -u -m src.eval.probing.probe_phoneme --models $MODEL
echo "Phoneme probe evaluation completed."

# echo "Running speaker probe"
# python -u -m src.eval.probing.probe_speaker --models $MODEL
# echo "Speaker probe evaluation completed."


echo "Accent probe evaluation completed."