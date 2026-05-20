#!/usr/bin/env python3
"""
Prepare utterance mappings for E2 Latent Diffusion Bridge training.

Builds:
  - utterance_mapping.json: (L2_utt, CMU_utt) pairs for bridge training, split 85/15 by L1
  - eval_mapping.json: Evaluation set (L2 test speakers + BDL native sanity check)

Usage:
  python prepare_data.py --output_dir src/experiments/exp2_latent_diffusion_bridge/data
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import soundfile
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from src.utils.load_l2arctic import (
    load_train_dev_utterances,
    load_test_utterances,
    _load_cmu_arctic_utterances,
    CMU_SPEAKERS,
)
from src.config import TEST_SPEAKERS


def extract_prompt_id(utterance_id: str) -> str:
    """
    Safe extraction of prompt ID from utterance_id.
    Handles: "ABA_arctic_a0001", "arctic_a0001", etc.
    Returns: "arctic_a0001"
    """
    idx = utterance_id.find("arctic_")
    if idx == -1:
        raise ValueError(f"Cannot extract prompt ID from: {utterance_id}")
    return utterance_id[idx:]


def get_encoder_state_path(speaker: str, utterance_id: str, data_dir: Path) -> Path | None:
    """
    Find the encoder state .pt file for a given speaker and utterance.
    Files are named {speaker}_{utterance_id}.pt in {data_dir}/{speaker}/ directory.
    """
    speaker_dir = data_dir / speaker
    if not speaker_dir.exists():
        return None

    # Construct filename with speaker prefix: {speaker}_{utterance_id}.pt
    # (utterance_id may or may not already include speaker prefix)
    if utterance_id.startswith(f"{speaker}_"):
        # utterance_id already has speaker prefix
        pt_file = speaker_dir / f"{utterance_id}.pt"
    else:
        # Add speaker prefix to utterance_id
        pt_file = speaker_dir / f"{speaker}_{utterance_id}.pt"

    if pt_file.exists():
        return pt_file
    return None


def build_prompt_to_utterances(utterances: List[Dict]) -> Dict[str, List[Dict]]:
    """Group utterances by prompt_id."""
    mapping = defaultdict(list)
    for utt in utterances:
        try:
            prompt_id = extract_prompt_id(utt["utterance_id"])
            mapping[prompt_id].append(utt)
        except ValueError as e:
            print(f"[WARN] Skipping utterance: {e}")
    return mapping


def compute_speech_end_frame(wav_path: str, padding: int = 20) -> int:
    """
    Compute speech_end_frame from wav duration.
    speech_end_frame = min(1500, int(duration_s * 50)) + padding
    """
    try:
        duration_s = soundfile.info(wav_path).duration
        frame = min(1500, int(duration_s * 50))
        return min(1500, frame + padding)  # Cap at 1500
    except Exception as e:
        print(f"[ERROR] Failed to get duration for {wav_path}: {e}")
        raise


def build_training_pairs(
    l2_utterances: List[Dict],
    cmu_speakers: List[str],
    cmu_prompt_map: Dict[str, List[Dict]],
    data_dirs: Dict[str, Path],
) -> List[Dict]:
    """
    Build (L2_utterance, CMU_utterance) pairs by matching prompt IDs.
    For each L2 utterance, creates 3 pairs (one per CMU speaker: CLB, RMS, SLT).
    """
    pairs = []
    skipped = defaultdict(int)

    for l2_utt in tqdm(l2_utterances, desc="Building pairs"):
        try:
            prompt_id = extract_prompt_id(l2_utt["utterance_id"])
        except ValueError:
            skipped["bad_prompt_id"] += 1
            continue

        # Check if CMU has this prompt
        if prompt_id not in cmu_prompt_map:
            skipped["no_matching_prompt"] += 1
            continue

        # Get L2 encoder state path (search both train and dev)
        l2_path = None
        l2_split = None
        for split in ["train", "dev"]:
            if split in data_dirs:
                p = get_encoder_state_path(
                    l2_utt["speaker"],
                    l2_utt["utterance_id"],
                    data_dirs[split]
                )
                if p:
                    l2_path = p
                    l2_split = split
                    break

        if not l2_path:
            skipped[f"missing_l2_file"] += 1
            continue

        # Compute speech_end_frame from L2 wav
        try:
            speech_end_frame = compute_speech_end_frame(l2_utt["wav_path"])
        except Exception:
            skipped["duration_error"] += 1
            continue

        # Create pair for each CMU speaker
        for cmu_utt in cmu_prompt_map[prompt_id]:
            cmu_speaker = cmu_utt["speaker"]
            if cmu_speaker not in cmu_speakers:
                continue

            # Get CMU encoder state path (search both train and dev)
            cmu_path = None
            cmu_split = None
            for split in ["train", "dev"]:
                if split in data_dirs:
                    p = get_encoder_state_path(
                        cmu_speaker,
                        cmu_utt["utterance_id"],
                        data_dirs[split]
                    )
                    if p:
                        cmu_path = p
                        cmu_split = split
                        break

            if not cmu_path:
                skipped[f"missing_cmu_file_{cmu_speaker}"] += 1
                continue

            # Save paths relative to their split directory for cross-environment portability
            l2_rel_path = l2_path.relative_to(data_dirs[l2_split])
            cmu_rel_path = cmu_path.relative_to(data_dirs[cmu_split])

            pair = {
                "prompt_id": prompt_id,
                "l2_speaker": l2_utt["speaker"],
                "l2_utterance_id": l2_utt["utterance_id"],
                "l2_encoder_state_path": str(l2_rel_path),
                "nat_speaker": cmu_speaker,
                "nat_utterance_id": cmu_utt["utterance_id"],
                "nat_encoder_state_path": str(cmu_rel_path),
                "speech_end_frame": speech_end_frame,
                "text": l2_utt.get("text", ""),
                "l1": l2_utt.get("l1", "Unknown"),
                "bridge_split": "train",  # Will be reassigned during stratified split
            }
            pairs.append(pair)

    print("\n[Pairing Stats]")
    print(f"  Total pairs created: {len(pairs)}")
    for reason, count in sorted(skipped.items()):
        print(f"  Skipped ({reason}): {count}")

    return pairs


def stratified_split(
    pairs: List[Dict],
    train_ratio: float = 0.85,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Split pairs 85/15 stratified by L1.
    Modifies pairs in-place by setting bridge_split field.
    """
    # Group by L1
    by_l1 = defaultdict(list)
    for pair in pairs:
        by_l1[pair["l1"]].append(pair)

    print(f"\n[L1 Stratification]")
    for l1 in sorted(by_l1.keys()):
        print(f"  {l1}: {len(by_l1[l1])} pairs")

    # Split each L1 group
    train_indices = set()
    for l1, l1_pairs in by_l1.items():
        l1_indices = [pairs.index(p) for p in l1_pairs]
        train_idx, _ = train_test_split(
            l1_indices,
            train_size=train_ratio,
            random_state=random_seed,
        )
        train_indices.update(train_idx)

    # Assign splits
    for i, pair in enumerate(pairs):
        pair["bridge_split"] = "train" if i in train_indices else "val"

    train_count = sum(1 for p in pairs if p["bridge_split"] == "train")
    val_count = sum(1 for p in pairs if p["bridge_split"] == "val")
    print(f"\n[Split]")
    print(f"  Train: {train_count} pairs ({100*train_count/len(pairs):.1f}%)")
    print(f"  Val:   {val_count} pairs ({100*val_count/len(pairs):.1f}%)")

    return pairs


def build_eval_set(
    l2_test_utterances: List[Dict],
    cmu_test_utterances: List[Dict],
) -> List[Dict]:
    """
    Build evaluation metadata for test utterances.
    Encoder state paths are reconstructed at eval time via TEST_DATA_DIR env var.
    """
    eval_pairs = []

    # L2 test speakers
    for l2_utt in tqdm(l2_test_utterances, desc="Building eval set (L2 test)"):
        try:
            speech_end_frame = compute_speech_end_frame(l2_utt["wav_path"])
        except Exception:
            speech_end_frame = None
        eval_pairs.append({
            "speaker": l2_utt["speaker"],
            "utterance_id": l2_utt["utterance_id"],
            "text": l2_utt.get("text", ""),
            "l1": l2_utt.get("l1", "Unknown"),
            "eval_type": "l2_test",
            "speech_end_frame": speech_end_frame,
        })

    # CMU test speakers (native sanity check)
    for cmu_utt in tqdm(cmu_test_utterances, desc="Building eval set (CMU sanity check)"):
        eval_pairs.append({
            "speaker": cmu_utt["speaker"],
            "utterance_id": cmu_utt["utterance_id"],
            "text": cmu_utt.get("text", ""),
            "l1": "English",
            "eval_type": "native_sanity_check",
        })

    print(f"\n[Eval Set]")
    print(f"  L2 test speakers: {sum(1 for p in eval_pairs if p['eval_type'] == 'l2_test')} utterances")
    print(f"  BDL native check: {sum(1 for p in eval_pairs if p['eval_type'] == 'native_sanity_check')} utterances")
    print(f"  Total eval: {len(eval_pairs)} utterances")

    return eval_pairs


def prepare_data(
    output_dir: str = "src/experiments/exp2_latent_diffusion_bridge/data",
    train_data_dir: str = "data/processed/train",
    dev_data_dir: str = "data/processed/dev",
    test_data_dir: str = "data/processed/test",
    test_only: bool = False,
):
    """Main entry point for data preparation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder state directories
    data_dirs = {}
    for split_name, split_dir_str in [("train", train_data_dir), ("dev", dev_data_dir), ("test", test_data_dir)]:
        split_dir = Path(split_dir_str)
        if split_dir.exists():
            data_dirs[split_name] = split_dir

    if not data_dirs:
        raise RuntimeError(f"No processed data directories found. Check paths: train={train_data_dir}, dev={dev_data_dir}, test={test_data_dir}")

    print(f"[Data Directories]")
    for split, path in sorted(data_dirs.items()):
        print(f"  {split}: {path}")

    print(f"\n[Loading Utterances]")
    l2_test = load_test_utterances()
    print(f"  L2 test: {len(l2_test)} utterances")

    cmu_test_speakers = CMU_SPEAKERS & TEST_SPEAKERS
    cmu_test_utterances = _load_cmu_arctic_utterances(speakers=cmu_test_speakers) if cmu_test_speakers else []
    if cmu_test_utterances:
        print(f"  CMU test speakers {sorted(cmu_test_speakers)}: {len(cmu_test_utterances)} utterances")

    test_pairs = build_eval_set(l2_test, cmu_test_utterances)
    test_mapping_path = output_dir / "mapping_test.json"
    with open(test_mapping_path, "w") as f:
        json.dump(test_pairs, f, indent=2)
    print(f"[Saved] {test_mapping_path}")

    if test_only:
        print(f"\n[✅ Test mapping updated]")
        return

    l2_train, l2_dev = load_train_dev_utterances()
    print(f"  L2 train: {len(l2_train)} utterances")
    print(f"  L2 dev: {len(l2_dev)} utterances")

    cmu_train_speakers = CMU_SPEAKERS - TEST_SPEAKERS
    cmu_utterances = [u for u in l2_train + l2_dev if u["speaker"] in cmu_train_speakers]
    cmu_prompt_map = build_prompt_to_utterances(cmu_utterances)
    print(f"  CMU train speakers: {sorted(cmu_train_speakers)}")
    print(f"    {len(cmu_utterances)} utterances across {len(cmu_prompt_map)} prompts")

    l2_combined = [u for u in l2_train + l2_dev if u["speaker"] not in cmu_train_speakers]
    print(f"  L2 utterances (after filtering CMU): {len(l2_combined)}")
    pairs = build_training_pairs(l2_combined, cmu_train_speakers, cmu_prompt_map, data_dirs)
    pairs = stratified_split(pairs, train_ratio=0.85)

    train_pairs = [p for p in pairs if p["bridge_split"] == "train"]
    dev_pairs   = [p for p in pairs if p["bridge_split"] == "val"]

    train_mapping_path = output_dir / "mapping_train.json"
    dev_mapping_path   = output_dir / "mapping_dev.json"
    with open(train_mapping_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(dev_mapping_path, "w") as f:
        json.dump(dev_pairs, f, indent=2)

    print(f"[Saved] {train_mapping_path}")
    print(f"[Saved] {dev_mapping_path}")
    print(f"\n[✅ Data preparation complete]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare utterance mappings for E2 Latent Diffusion Bridge"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data",
        help="Output directory for JSON mappings",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data/processed/train",
        help="Directory containing training encoder states",
    )
    parser.add_argument(
        "--dev_data_dir",
        type=str,
        default="data/processed/dev",
        help="Directory containing dev encoder states",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="data/processed/test",
        help="Directory containing test encoder states",
    )

    parser.add_argument("--test_only", action="store_true",
                        help="Only regenerate mapping_test.json (preserves train/dev mappings)")
    args = parser.parse_args()

    prepare_data(
        output_dir=args.output_dir,
        train_data_dir=args.train_data_dir,
        dev_data_dir=args.dev_data_dir,
        test_data_dir=args.test_data_dir,
        test_only=args.test_only,
    )
