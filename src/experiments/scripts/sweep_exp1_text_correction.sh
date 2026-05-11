#!/bin/bash
# Sweep script for Experiment 1: submit all 6 configs
# SLURM will queue them automatically (max 3 A30s used)
#
# Usage: bash src/experiments/scripts/sweep_exp1_text_correction.sh

set -e

PROJECT_ROOT=/rds/general/user/tsv22/home/accent-robust-asr
cd $PROJECT_ROOT

mkdir -p logs

CONFIGS=(
  "low_perturb_high_masking.json"
  "low_perturb_wide_masking.json"
  "medium_perturb_medium_masking.json"
  "max_perturb_high_masking.json"
  "max_perturb_medium_masking.json"
  "phoneme_perturb_low.json"
)

echo "========================================"
echo "Submitting Experiment 1 sweep (6 configs)"
echo "SLURM will queue them automatically"
echo "========================================"
echo ""

JOB_IDS=()
for CONFIG in "${CONFIGS[@]}"; do
  echo "Submitting $CONFIG..."
  JOB_ID=$(sbatch \
    --job-name=exp1_${CONFIG%.json} \
    --partition=a30 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --time=0:30:00 \
    --output=logs/exp1_${CONFIG%.json}_%j.out \
    --error=logs/exp1_${CONFIG%.json}_%j.out \
    --wrap="cd $PROJECT_ROOT && python -u -m src.experiments.exp1_text_correction.train --config src/experiments/exp1_text_correction/configs/$CONFIG --device cuda" \
    | grep -oP '(?<=Submitted batch job )\d+')
  JOB_IDS+=($JOB_ID)
  echo "  Job ID: $JOB_ID"
done

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u tsv22"
echo ""
echo "View results when complete:"
echo "  for dir in results/experiment1_stage1/*/; do"
echo "    echo \"=== \$(basename \$dir) ===\""
echo "    tail -3 \$dir/results.csv"
echo "  done"
echo "========================================"
