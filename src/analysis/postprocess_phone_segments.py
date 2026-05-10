"""
Post-process encoder states to extract phone-segment mean pools.

Loads layer_outputs from extraction, parses TextGrids/labs, pools frames within phone spans.
Works identically for L2-ARCTIC (TextGrid) and CMU-ARCTIC (lab) speakers.

Inputs:
  - $EPHEMERAL/accent-robust-asr/probing/encoder_states/<split>/<SPEAKER>/<utterance_id>.pt
  - L2-ARCTIC: data/l2_arctic/<SPEAKER>/textgrid/*.TextGrid
  - CMU: data/cmu_arctic/cmu_us_<speaker>_arctic/lab/*.lab

Outputs:
  $EPHEMERAL/accent-robust-asr/probing/phone_segments/<split>/<SPEAKER>/<utterance_id>_phones.pkl
    Dict with keys:
      - phone_reps: list[dict] with keys label, l1, speaker, layer_reps (12, 768)
"""

import torch
import pickle
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import numpy as np

from src.config import SPEAKER_L1, ENCODER_FRAME_RATE
from src.utils.load_l2arctic import load_train_dev_utterances, load_test_utterances
from src.utils.textgrid import parse_textgrid


# Lab file parsing for CMU ARCTIC
SILENCE_LABELS_LAB = {'pau', 'sp', 'brth', '#', ''}
LAB_LABEL_MAP = {'ax': 'AH', 'ix': 'IH', 'ax-h': 'AH', 'axr': 'ER', 'el': 'L', 'em': 'M', 'en': 'N'}


def parse_lab_file(lab_path: str) -> List[tuple]:
    """Parse Festival/HTK .lab file → list of (start_sec, end_sec, phone_label)."""
    segments = []
    prev_end = 0.0
    try:
        with open(lab_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                end_time = float(parts[0])
                label = parts[2]
                if label in SILENCE_LABELS_LAB:
                    prev_end = end_time
                    continue
                label_norm = LAB_LABEL_MAP.get(label, label).upper()
                segments.append((prev_end, end_time, label_norm))
                prev_end = end_time
    except Exception as e:
        print(f"  [WARN] Failed to parse {lab_path}: {e}")
    return segments


def extract_phone_reps_from_layer_outputs(
    layer_outputs: torch.Tensor,
    phone_segments: List[tuple],
    speaker: str,
    frame_rate: int = 50,
) -> List[Dict]:
    """
    Extract mean-pooled phone representations from layer outputs.

    Args:
        layer_outputs: (12, T, 768) tensor
        phone_segments: list of (start_sec, end_sec, label) tuples
        speaker: speaker ID (for metadata)
        frame_rate: 50 Hz for Whisper

    Returns:
        list[dict] with keys: label, l1, speaker, layer_reps (12, 768)
    """
    l1 = SPEAKER_L1.get(speaker, "Unknown")
    phone_reps = []
    T = layer_outputs.shape[1]

    for start_sec, end_sec, label in phone_segments:
        sf = max(0, int(start_sec * frame_rate))
        ef = max(sf + 1, min(int(end_sec * frame_rate), T))

        # Mean-pool across frames for all 12 layers
        segment_reps = layer_outputs[:, sf:ef, :].mean(dim=1)  # (12, 768)

        phone_reps.append({
            "label": label,
            "l1": l1,
            "speaker": speaker,
            "layer_reps": segment_reps.cpu().numpy(),  # (12, 768) float32
        })

    return phone_reps


def is_already_processed(utt: Dict, split_output_dir: Path) -> bool:
    """Check if utterance has already been processed."""
    output_path = split_output_dir / utt["speaker"] / f"{utt['utterance_id']}_phones.pkl"
    return output_path.exists()


def process_utterance(
    utt: Dict,
    encoder_states_dir: Path,
    output_dir: Path,
    l2arctic_root: Path,
    cmu_arctic_root: Path,
) -> bool:
    """
    Load encoder states, parse phone alignment, extract phone reps, save.

    Returns True if successful, False otherwise.
    """
    try:
        # Load encoder states
        encoder_path = encoder_states_dir / utt["speaker"] / f"{utt['utterance_id']}.pt"
        if not encoder_path.exists():
            print(f"  [WARN] Encoder states not found: {encoder_path}")
            return False

        state = torch.load(encoder_path, weights_only=False)
        layer_outputs = state.get("layer_outputs")
        if layer_outputs is None:
            print(f"  [WARN] No layer_outputs in {encoder_path}")
            return False

        # Parse phone alignment
        phone_segments = []
        if utt.get("textgrid"):
            # L2-ARCTIC: parse TextGrid
            try:
                textgrid_segs = parse_textgrid(utt["textgrid"], tier_name="phones")
                # Convert PhoneSegment objects to (start_sec, end_sec, label) tuples
                phone_segments = [(seg.start, seg.end, seg.label) for seg in textgrid_segs]
            except Exception as e:
                print(f"  [WARN] Failed to parse TextGrid {utt['textgrid']}: {e}")
                return False
        else:
            # CMU: parse .lab file
            speaker_lower = utt["speaker"].lower()
            lab_path = cmu_arctic_root / f"cmu_us_{speaker_lower}_arctic" / "lab" / f"{utt['utterance_id']}.lab"
            if not lab_path.exists():
                print(f"  [WARN] Lab file not found: {lab_path}")
                return False
            phone_segments = parse_lab_file(str(lab_path))

        if not phone_segments:
            print(f"  [WARN] No phone segments extracted: {utt['utterance_id']}")
            return False

        # Extract phone reps
        phone_reps = extract_phone_reps_from_layer_outputs(
            layer_outputs, phone_segments, utt["speaker"]
        )

        # Save
        speaker_dir = output_dir / utt["speaker"]
        speaker_dir.mkdir(parents=True, exist_ok=True)

        output_path = speaker_dir / f"{utt['utterance_id']}_phones.pkl"
        temp_path = speaker_dir / f".{utt['utterance_id']}_phones.pkl.tmp"

        with open(temp_path, 'wb') as f:
            pickle.dump({"phone_reps": phone_reps}, f)

        temp_path.rename(output_path)
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {utt['utterance_id']}: {e}")
        return False


def main(encoder_dir: str = None, output_dir: str = None):
    """Post-process all utterances: extract phone segments from encoder states."""

    # Set up paths
    if encoder_dir is None:
        encoder_dir = os.environ.get("EPHEMERAL", "data/processed")
    encoder_base = Path(encoder_dir) / "accent-robust-asr" / "probing" / "encoder_states"

    if output_dir is None:
        output_dir = os.environ.get("EPHEMERAL", "data/processed")
    output_base = Path(output_dir) / "accent-robust-asr" / "probing" / "phone_segments"

    encoder_base.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    l2arctic_root = Path("data/l2_arctic")
    cmu_arctic_root = Path("data/cmu_arctic")

    print(f"[postprocess_phone_segments] Encoder states: {encoder_base}")
    print(f"[postprocess_phone_segments] Output: {output_base}")

    # Load all utterances
    train, dev = load_train_dev_utterances()
    test = load_test_utterances()
    all_utts = train + dev + test
    print(f"[postprocess_phone_segments] Processing {len(all_utts)} utterances")

    splits = {"train": train, "dev": dev, "test": test}

    total_success = 0
    total_failed = 0
    total_skipped = 0

    for split_name, utterances in splits.items():
        print(f"\n[postprocess_phone_segments] {split_name} split ({len(utterances)} utterances)...")
        split_output_dir = output_base / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        split_encoder_dir = encoder_base / split_name

        if not split_encoder_dir.exists():
            print(f"  [WARN] Encoder dir not found: {split_encoder_dir}")
            continue

        # Clean up leftover temp files from interrupted runs
        if split_output_dir.exists():
            for tmp_file in split_output_dir.glob("**/.*.pkl.tmp"):
                tmp_file.unlink()

        # Filter out already-processed utterances
        remaining = [u for u in utterances if not is_already_processed(u, split_output_dir)]
        skipped = len(utterances) - len(remaining)
        if skipped > 0:
            print(f"  [{skipped} already processed, resuming from {len(remaining)} remaining]")
            total_skipped += skipped

        for utt in tqdm(remaining, desc=f"  {split_name}"):
            if process_utterance(utt, split_encoder_dir, split_output_dir, l2arctic_root, cmu_arctic_root):
                total_success += 1
            else:
                total_failed += 1

    print(f"\n[postprocess_phone_segments] Done!")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Output: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process encoder states to extract phone segments")
    parser.add_argument("--encoder_dir", type=str, default=None,
                       help="Encoder states directory (default: uses $EPHEMERAL if set)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: uses $EPHEMERAL if set)")
    args = parser.parse_args()
    main(encoder_dir=args.encoder_dir, output_dir=args.output_dir)
