#!/bin/bash
# Experiment 1 hyperparameter sweep: submit all 6 configs as separate jobs
#
# Usage: bash src/experiments/scripts/sweep_exp1_text_correction.sh

set -e

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

mkdir -p logs

CONFIGS=(
  "low_perturb_high_masking.json"
  "low_perturb_wide_masking.json"
  "medium_perturb_medium_masking.json"
  "max_perturb_high_masking.json"
  "max_perturb_medium_masking.json"
  "low_perturb_medium_masking.json"
)

echo "========================================"
echo "Submitting Experiment 1 sweep (6 configs)"
echo "========================================"
echo ""

JOB_IDS=()
for CONFIG in "${CONFIGS[@]}"; do
  CONFIG_PATH="src/experiments/exp1_text_correction/configs/$CONFIG"
  echo "Submitting $CONFIG..."
  JOB_ID=$(sbatch \
    --job-name="exp1_${CONFIG%.json}" \
    src/experiments/scripts/slurm/train_exp1_text_correction.sh \
    "$CONFIG_PATH" \
    | grep -oP 'Submitted batch job \K\d+')
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
