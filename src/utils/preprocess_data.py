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
from tqdm import tqdm
import shutil

from src.config import SPEAKER_L1
from src.utils.load_l2arctic import load_test_utterances, load_train_dev_utterances

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
    splits: list[str] = ["scripted"],
    whisper_model: str = "openai/whisper-small",
    held_out_l1: str = None
):
    
    for split in splits:
        """Step 1: Create LibriSpeech-like raw structure → Step 2: Whisper encode."""
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Stage 1: Create raw/ structure matching LibriSpeech
        ho_suf = "" if held_out_l1 is None else f"heldout_{held_out_l1}"
        train_split_dir = {
            "train" : f"train_{ho_suf}",
            "dev": f"dev_{ho_suf}",
            "test": "test"
        }

        test_utts = load_test_utterances(split=split)
        train_utts, dev_utts = load_train_dev_utterances(held_out_l1=held_out_l1)


        for train_split in ["train", "dev", "test"]:

            if split == "spontaneous" and train_split != "test":
                print(f"Skipping {train_split} split for spontaneous data.")
                continue

            raw_split_dir = Path(raw_output_dir) / train_split_dir[train_split] / split
            if not (raw_split_dir.exists() and any(raw_split_dir.iterdir())):
                print(f"Creating raw directory for {train_split} split: {raw_split_dir}")
                raw_split_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Raw directory for {train_split} split already exists and is not empty: {raw_split_dir}, skipping creation.")
                continue
            utts = {"train": train_utts, "dev": dev_utts, "test": test_utts}[train_split]
            
            # Copy WAVs + create fake .trans.txt (LibriSpeech format)
            transcripts = {}
            
            for utt in tqdm(utts, desc="Copying WAVs"):
                speaker = utt["speaker"]
                utt_id = f"{speaker}_{utt['utterance_id']}"

                speaker_dir = raw_split_dir / speaker
                speaker_dir.mkdir(parents=True, exist_ok=True)

                wav_dst = speaker_dir / f"{utt_id}.wav"
                shutil.copy2(utt["wav_path"], wav_dst)

                transcripts[utt_id] = utt["text"]
        
            # Write single .trans.txt (LibriSpeech format: "utt_id text")
            with open(raw_split_dir / "data.trans.txt", "w") as f:
                for utt_id, text in sorted(transcripts.items()):
                    f.write(f"{utt_id} {text}\n")
            
            print(f"✅ Raw data: {raw_split_dir}")
            print(f"Created {len(transcripts)} WAVs + data.trans.txt")
        
            # Stage 2: Run Whisfusion preprocess_audio on our fake LibriSpeech
            processed_dir = Path(processed_output_dir) / train_split_dir[train_split] / split

            if processed_dir.exists() and any(processed_dir.iterdir()):
                print(f"Processed directory already exists and is not empty: {processed_dir}, skipping preprocessing.")
                continue

            processed_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing with Whisper encoder...")
            
            import subprocess
            subprocess.run([
                "python",
                "-m",
                "models.whisfusion.src.data.preprocess_audio",
                "--source_dir", str(raw_split_dir),
                "--output_dir", str(processed_dir),
                "--model_name", whisper_model
            ], check=True)
            
            print(f"\n✅ Processed .pt files: {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess L2-Arctic for Whisfusion")
    parser.add_argument("--held_out_l1", default=None, help="Hold out this L1 when creating the files")
    parser.add_argument("--raw_output_dir", type=str, default="data/raw", help="Whisfusion raw dir")
    parser.add_argument("--processed_output_dir", type=str, default="data/processed", help="Whisfusion processed dir")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["scripted"],
        choices=["scripted", "spontaneous"],
        help="One or more splits to process (default: scripted). Example: --split scripted spontaneous",
    )
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")
    
    args = parser.parse_args()
    preprocess_l2arctic(args.raw_output_dir, args.processed_output_dir, args.split, args.whisper_model, args.held_out_l1)