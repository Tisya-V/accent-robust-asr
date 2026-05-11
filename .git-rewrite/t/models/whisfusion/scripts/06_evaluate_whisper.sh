#!/bin/bash
# Evaluates Whisper models on LibriSpeech test sets
#
# Usage:
# 1. chmod +x scripts/06_evaluate_whisper.sh
# 2. ./scripts/06_evaluate_whisper.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
OUTPUT_BASE_DIR="./evaluation_results/whisper"
NUM_RUNS=5
MAX_FILES=""

# Create output directories
mkdir -p ${OUTPUT_BASE_DIR}/test-clean
mkdir -p ${OUTPUT_BASE_DIR}/test-other

# Models to evaluate
MODELS=(
   "openai/whisper-tiny"
   "openai/whisper-small"
   "openai/whisper-large-v3-turbo"
)

print_status() {
   echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
   echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
   echo -e "${RED}[✗]${NC} $1"
}

print_header() {
   echo -e "\n${YELLOW}========================================${NC}"
   echo -e "${YELLOW}$1${NC}"
   echo -e "${YELLOW}========================================${NC}\n"
}

run_evaluation() {
   local dataset=$1
   local output_dir="${OUTPUT_BASE_DIR}/${dataset}"
   
   print_header "Evaluating on LibriSpeech ${dataset}"
   
   models_arg=""
   for model in "${MODELS[@]}"; do
       models_arg="${models_arg} ${model}"
   done
   
   cmd="python -m src.evaluation.evaluate_whisper \
       --data_path data/raw/LibriSpeech/${dataset} \
       --output_dir ${output_dir} \
       --models ${models_arg} \
       --num_runs ${NUM_RUNS}"
   
   if [ ! -z "${MAX_FILES}" ]; then
       cmd="${cmd} --max_files ${MAX_FILES}"
   fi
   
   print_status "Running evaluation with ${#MODELS[@]} models..."
   print_status "Command: ${cmd}"
   
   if eval ${cmd}; then
       print_success "Evaluation completed for ${dataset}"
   else
       print_error "Evaluation failed for ${dataset}"
       return 1
   fi
}

START_TIME=$(date +%s)

print_header "WHISPER MODELS EVALUATION"
print_status "Starting comprehensive evaluation..."
print_status "Models to test: ${#MODELS[@]}"
print_status "Output directory: ${OUTPUT_BASE_DIR}"
print_status "Runs per file: ${NUM_RUNS}"

run_evaluation "test-clean"
run_evaluation "test-other"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "EVALUATION COMPLETE"
print_success "All evaluations completed successfully!"
print_status "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
print_status "Results saved in: ${OUTPUT_BASE_DIR}"

echo -e "\n${BLUE}Output files:${NC}"
find ${OUTPUT_BASE_DIR} -name "evaluate_*.json" -type f | sort