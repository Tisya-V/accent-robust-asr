#!/bin/bash

#PBS -N e2_steering_experiment
#PBS -l walltime=3:00:00
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -j oe
#PBS -o logs/e2_steering_experiment.log

set -e

# Project root
PROJECT_ROOT="/rds/general/user/tsv22/home/accent-robust-asr"
cd "$PROJECT_ROOT"

# Load environment
source scripts/env.sh

# Create log directory
mkdir -p logs

# Run the steering experiment with output streaming
echo "=========================================="
echo "E2 Latent Steering Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $PROJECT_ROOT/src/analysis/results/e2_steering"
echo ""

python -m src.analysis.e2_latent_steering \
    --alpha_values 0.0 0.1 0.25 0.5 0.75 1.0 \
    --output_dir "$PROJECT_ROOT/src/analysis/results/e2_steering" \
    --device cuda 2>&1 | tee -a logs/e2_steering_experiment.log

echo ""
echo "End time: $(date)"
echo "✓ Experiment complete!"
