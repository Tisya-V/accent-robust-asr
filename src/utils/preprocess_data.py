#!/usr/bin/env python3
"""
Preprocess L2-Arctic for Whisfusion evaluation.
Creates data/raw/L2Arctic/test-clean/ with .wav + structure matching LibriSpeech.
Then runs Whisper encoder → data/processed/L2Arctic/test-clean/*.pt files.
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm
from typing import List, Dict
import shutil

# Your L2-Arctic loader (adapt paths)
from src.utils.load_l2arctic import load_test_utterances

def ensure_mono_and_resample(waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Convert to mono 16kHz."""
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze(0)

def preprocess_l2arctic(
    raw_output_dir: str,
    processed_output_dir: str,
    split: str = "scripted",
    whisper_model: str = "openai/whisper-small"
):
    """Step 1: Create LibriSpeech-like raw structure → Step 2: Whisper encode."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Stage 1: Create raw/ structure matching LibriSpeech
    raw_test_dir = Path(raw_output_dir) / split
    if not (raw_test_dir.exists() and any(raw_test_dir.iterdir())):
        # Load L2-Arctic test utterances
        print("Loading test set...")
        test_utts = load_test_utterances(split=split)
        print(f"Loaded {len(test_utts)} test utterances")

        raw_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy WAVs + create fake .trans.txt (LibriSpeech format)
        transcripts = {}
        for utt in tqdm(test_utts, desc="Copying WAVs"):
            speaker = utt["speaker"]  
            utt_id = f"{speaker}_{utt['utterance_id']}"  # e.g. "ABA_arctic_a0001"
            wav_src = Path(utt["wav_path"])
            wav_dst = raw_test_dir / f"{utt_id}.wav"
            
            # Copy WAV
            shutil.copy2(wav_src, wav_dst)
            
            # Store transcript for .trans.txt
            transcripts[utt_id] = utt["text"]
    
        # Write single .trans.txt (LibriSpeech format: "utt_id text")
        with open(raw_test_dir / "data.trans.txt", "w") as f:
            for utt_id, text in sorted(transcripts.items()):
                f.write(f"{utt_id} {text}\n")
        
        print(f"✅ Raw data: {raw_test_dir}")
        print(f"Created {len(transcripts)} WAVs + data.trans.txt")
    else:
        print(f"Raw test directory already exists: {raw_test_dir}, skipping creation.")
    
    # Stage 2: Run Whisfusion preprocess_audio on our fake LibriSpeech
    processed_dir = Path(processed_output_dir) / split

    if processed_dir.exists() and any(processed_dir.iterdir()):
        print(f"Processed directory already exists and is not empty: {processed_dir}, skipping preprocessing.")
        print(f"Run eval with: --data_path {raw_test_dir} or delete {processed_dir} to re-run preprocessing.")
        return

    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing with Whisper encoder...")
    
    import subprocess
    subprocess.run([
        "python",
        "-m",
        "models.whisfusion.src.data.preprocess_audio",
        "--source_dir", str(raw_test_dir),
        "--output_dir", str(processed_dir),
        "--model_name", whisper_model
    ], check=True)
    
    print(f"\n✅ Processed .pt files: {processed_dir}")
    print(f"Run eval with: --data_path {raw_test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess L2-Arctic for Whisfusion")
    parser.add_argument("--raw_output_dir", type=str, default="data/raw", help="Whisfusion raw dir")
    parser.add_argument("--processed_output_dir", type=str, default="data/processed", help="Whisfusion processed dir")
    parser.add_argument("--split", type=str, default="scripted", help="Split to process (scripted (default)/spontaneous)")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")
    
    args = parser.parse_args()
    preprocess_l2arctic(args.raw_output_dir, args.processed_output_dir, args.split, args.whisper_model)