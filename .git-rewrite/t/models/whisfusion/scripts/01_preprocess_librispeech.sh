#!/bin/bash
# Preprocesses LibriSpeech partitions through Whisper encoder and saves hidden states as .pt files
#
# Usage:
# 1. chmod +x scripts/01_preprocess_librispeech.sh
# 2. ./scripts/01_preprocess_librispeech.sh

set -e

# Configuration
RAW_DATA_ROOT="data/raw/LibriSpeech"
PROCESSED_DATA_ROOT="data/processed"

PARTITIONS=(
   "train-clean-100"
   "train-clean-360"
   "train-other-500"
   "dev-clean"
   "dev-other"
   "test-clean"
   "test-other"
)

echo "============================================================"
echo "Starting audio preprocessing for all partitions..."
echo "============================================================"

for part in "${PARTITIONS[@]}"; do
   INPUT_DIR="${RAW_DATA_ROOT}/${part}"
   OUTPUT_DIR="${PROCESSED_DATA_ROOT}/${part}"
   
   echo -e "\n--- Processing partition: ${part} ---"
   echo "Input directory:  ${INPUT_DIR}"
   echo "Output directory: ${OUTPUT_DIR}"
   
   if [ ! -d "$INPUT_DIR" ]; then
       echo "Warning: Input directory ${INPUT_DIR} not found. Skipping."
       continue
   fi
   
   python -m src.data.preprocess_audio \
       --source_dir "$INPUT_DIR" \
       --output_dir "$OUTPUT_DIR" \
       --model_name "openai/whisper-small"
   
   echo "-> Successfully processed ${part}"
done

echo -e "\n============================================================"
echo "✅ All partitions have been preprocessed successfully!"
echo "============================================================"