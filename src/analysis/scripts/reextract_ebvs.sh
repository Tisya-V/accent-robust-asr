#!/bin/bash
#PBS -N reextract_ebvs
#PBS -l select=1:ngpus=1:ncpus=2:mem=16gb
#PBS -l walltime=00:30:00
#PBS -o logs/reextract_ebvs.out
#PBS -e logs/reextract_ebvs.err
#PBS -j oe

# Re-extract encoder states for EBVS (Spanish) speaker only
# Usage: qsub src/analysis/scripts/reextract_ebvs.sh
# This script OVERWRITES existing .pt files

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create log file
mkdir -p logs
RUNTIME_LOG="logs/reextract_ebvs_${PBS_JOBID}.log"
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Re-extracting Whisper encoder for EBVS"
echo "Real-time log: $RUNTIME_LOG"
echo "=========================================="
echo ""

nvidia-smi
echo ""
echo "Re-extracting Whisper encoder states for EBVS..."

python3 << 'EOF'
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.config import MODEL_ID
from src.utils.load_l2arctic import load_test_utterances

# Load model
print("Loading Whisper encoder...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = model.to("cuda")
model.eval()

# Load test utterances
utterances = load_test_utterances()
ebvs_utterances = [u for u in utterances if u['speaker'] == 'EBVS']

print(f"Found {len(ebvs_utterances)} EBVS utterances")

# Output directory
output_dir = Path("data/processed/test/EBVS")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract encoder states
extracted = 0
with torch.no_grad():
    for utt in tqdm(ebvs_utterances, desc="Extracting EBVS"):
        wav_path = utt['wav_path']
        utt_id = utt['utterance_id']

        try:
            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            audio_input = processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            audio_input = {k: v.to("cuda") for k, v in audio_input.items()}

            # Extract encoder outputs (using model.model.encoder)
            outputs = model.model.encoder(
                audio_input['input_features'],
                output_hidden_states=False,
                return_dict=True,
            )

            # Final encoder output matches data/processed format [seq_len, hidden_dim]
            final_hidden = outputs.last_hidden_state.squeeze(0).cpu()

            # Save with atomic write (temp → rename)
            save_path = output_dir / f"EBVS_{utt_id}.pt"
            temp_path = output_dir / f".EBVS_{utt_id}.pt.tmp"

            torch.save({'hidden_states': final_hidden}, temp_path)
            temp_path.rename(save_path)

            extracted += 1
        except Exception as e:
            print(f"Error processing {utt_id}: {e}")

print(f"✅ Extracted {extracted}/{len(ebvs_utterances)} EBVS encoder states to {output_dir}")
EOF

echo ""
echo "Verify extraction: ls data/processed/test/EBVS/*.pt | wc -l"
