"""
Extract all 12 Whisper encoder layer outputs + phone-segment mean pools.

Outputs: data/processed/probing/<split>/<SPEAKER>/<utterance_id>.pt
  Keys:
    - layer_outputs: (12, T, 768) float32 — all encoder layer outputs
    - phone_segments: list[dict] with keys:
        - label: str (e.g. "AE")
        - l1: str (e.g. "Arabic")
        - speaker: str
        - layer_reps: (12, 768) float32 — mean-pooled per layer across phone frames
"""

import torch
import torchaudio
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

from src.config import (
    MODEL_ID,
    SPEAKER_L1,
    ENCODER_FRAME_RATE,
    WHISPER_HIDDEN_DIM,
    WHISPER_N_ENCODER_LAYERS,
)
from src.utils.load_l2arctic import load_train_dev_utterances, load_test_utterances
from src.utils.textgrid import parse_textgrid


def load_audio(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr (16kHz)."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0).numpy()


def extract_encoder_outputs(
    audio_array: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
) -> torch.Tensor:
    """
    Extract all 12 encoder layer outputs for audio.

    Returns:
        torch.Tensor of shape (12, T, 768) where T is the time dimension.
    """
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        outputs = model.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )

    # outputs.hidden_states is a tuple of 13 tensors:
    # [0] = embedding output (after conv, before transformer)
    # [1..12] = transformer layer 1..12
    # Each shape: (1, T, 768)

    # Stack layers 1-12 (skip embedding, use transformer layers)
    layer_outputs = torch.stack([h.squeeze(0) for h in outputs.hidden_states[1:]], dim=0)
    return layer_outputs


def extract_phone_segment_reps(
    layer_outputs: torch.Tensor,  # (12, T, 768)
    textgrid_path: str,
    speaker: str,
) -> List[Dict]:
    """
    Extract mean-pooled representations for each phone segment.

    Args:
        layer_outputs: (12, T, 768)
        textgrid_path: path to forced-alignment TextGrid
        speaker: speaker ID

    Returns:
        List of dicts with keys: label, l1, speaker, layer_reps (12, 768)
    """
    try:
        phone_segments = parse_textgrid(textgrid_path, tier_name="phones")
    except Exception as e:
        print(f"  [WARN] Failed to parse TextGrid {textgrid_path}: {e}")
        return []

    l1 = SPEAKER_L1.get(speaker, "Unknown")
    phone_reps = []

    for seg in phone_segments:
        start_frame = seg.start_frame
        end_frame = seg.end_frame

        # Clamp to valid range
        T = layer_outputs.shape[1]
        start_frame = max(0, min(start_frame, T - 1))
        end_frame = max(start_frame + 1, min(end_frame, T))

        # Mean-pool across frames for each layer
        segment_reps = layer_outputs[:, start_frame:end_frame, :].mean(dim=1)  # (12, 768)

        phone_reps.append({
            "label": seg.label,
            "l1": l1,
            "speaker": speaker,
            "layer_reps": segment_reps.cpu().numpy(),  # (12, 768) float32
        })

    return phone_reps


def process_utterance(
    utt: Dict,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    output_dir: Path,
) -> bool:
    """
    Process a single utterance: extract encoder outputs + phone segments.
    Save to output_dir/<speaker>/<utterance_id>.pt atomically (temp file → rename).

    Returns True if successful, False otherwise.
    """
    try:
        audio_array = load_audio(utt["wav_path"])
        layer_outputs = extract_encoder_outputs(audio_array, model, processor, device)

        phone_reps = []
        if utt["textgrid"]:
            phone_reps = extract_phone_segment_reps(
                layer_outputs, utt["textgrid"], utt["speaker"]
            )

        speaker_dir = output_dir / utt["speaker"]
        speaker_dir.mkdir(parents=True, exist_ok=True)

        output_path = speaker_dir / f"{utt['utterance_id']}.pt"
        temp_path = speaker_dir / f".{utt['utterance_id']}.pt.tmp"

        torch.save({
            "layer_outputs": layer_outputs.cpu(),  # (12, T, 768)
            "phone_segments": phone_reps,
        }, temp_path)

        temp_path.rename(output_path)

        return True
    except Exception as e:
        print(f"  [ERROR] Failed to process {utt['utterance_id']}: {e}")
        return False


def is_already_processed(utt: Dict, split_output_dir: Path) -> bool:
    """Check if utterance has already been processed."""
    output_path = split_output_dir / utt["speaker"] / f"{utt['utterance_id']}.pt"
    return output_path.exists()


def main(output_dir: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_encoder_states] Using device: {device}")

    # Load model
    print(f"[extract_encoder_states] Loading {MODEL_ID}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, local_files_only=True
    ).to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    # Output directory (use EPHEMERAL if provided, else default to home dir)
    if output_dir is None:
        output_dir = os.environ.get("EPHEMERAL", "data/processed/probing")
    output_base = Path(output_dir) / "probing"
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"[extract_encoder_states] Output directory: {output_base}")

    # Load all utterances
    print("[extract_encoder_states] Loading train/dev/test utterances...")
    train, dev = load_train_dev_utterances()
    test = load_test_utterances()
    all_utts = train + dev + test
    print(f"  Total utterances: {len(all_utts)}")

    # Group by split for organized output
    splits = {"train": train, "dev": dev, "test": test}

    total_success = 0
    total_failed = 0
    total_skipped = 0

    with torch.no_grad():
        for split_name, utterances in splits.items():
            print(f"\n[extract_encoder_states] Processing {split_name} split ({len(utterances)} utterances)...")
            split_output_dir = output_base / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up leftover temp files from interrupted runs
            if split_output_dir.exists():
                for tmp_file in split_output_dir.glob("**/.*.pt.tmp"):
                    tmp_file.unlink()

            # Filter out already-processed utterances
            remaining = [u for u in utterances if not is_already_processed(u, split_output_dir)]
            skipped = len(utterances) - len(remaining)
            if skipped > 0:
                print(f"  [{skipped} already processed, resuming from {len(remaining)} remaining]")
                total_skipped += skipped

            for utt in tqdm(remaining, desc=f"  {split_name}"):
                if process_utterance(utt, model, processor, device, split_output_dir):
                    total_success += 1
                else:
                    total_failed += 1

    print(f"\n[extract_encoder_states] Done!")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Output directory: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Whisper encoder states for probing")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: uses $EPHEMERAL if set, else data/processed/probing)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
