#!/bin/bash
#SBATCH --job-name=exp1_profile
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Profile Experiment 1 training bottlenecks (20 batches)
#
# Usage:
# sbatch src/experiments/scripts/profile_exp1.sh
#
# Output will be printed to the job log (logs/exp1_profile_*.out)
# Look for "PROFILING SUMMARY" section

# Source centralized environment configuration
source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

set -e

echo "==============================="
echo "Profiling Experiment 1 (20 batches)"
echo "==============================="
echo ""

nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

echo ""
echo "Running profiler..."
python src/experiments/exp1_text_correction/profile_exp1.py

echo ""
echo "✅ Profiling complete."
echo "Check the output above for PROFILING SUMMARY table"
