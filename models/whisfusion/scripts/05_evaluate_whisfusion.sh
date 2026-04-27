#!/bin/bash
# Evaluates Whisfusion on LibriSpeech test sets
#
# Usage:
# 1. chmod +x scripts/05_evaluate_whisfusion.sh
# 2. ./scripts/05_evaluate_whisfusion.sh

set -e

# Configuration
BASE_MODEL_PATH="pretrained_models/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors"
ADAPTER_PATH="out/whisfusion_stage2_decoder.pt"
OUTPUT_DIR="./evaluation_results/whisfusion"

# Evaluation parameters
NUM_RUNS=5
N_CANDIDATES=15
N_STEPS=4

echo "============================================================"
echo "Starting Whisfusion evaluation on LibriSpeech test sets..."
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Evaluate on test-clean
echo -e "\n--- Evaluating on test-clean ---"
python -m src.evaluation.evaluate_whisfusion \
   --data_path data/raw/LibriSpeech/test-clean \
   --base_model_path "$BASE_MODEL_PATH" \
   --adapter_path "$ADAPTER_PATH" \
   --output_dir "$OUTPUT_DIR" \
   --num_runs "$NUM_RUNS" \
   --n_candidates "$N_CANDIDATES" \
   --n_steps "$N_STEPS"

# Evaluate on test-other
echo -e "\n--- Evaluating on test-other ---"
python -m src.evaluation.evaluate_whisfusion \
   --data_path data/raw/LibriSpeech/test-other \
   --base_model_path "$BASE_MODEL_PATH" \
   --adapter_path "$ADAPTER_PATH" \
   --output_dir "$OUTPUT_DIR" \
   --num_runs "$NUM_RUNS" \
   --n_candidates "$N_CANDIDATES" \
   --n_steps "$N_STEPS"

echo -e "\n============================================================"
echo "✅ Whisfusion evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "============================================================"