#!/usr/bin/env python3
"""
Preprocess unified L2-Arctic data for Whisfusion.
Treats scripted + spontaneous as one dataset.
Creates:
  data/raw/{train,dev,test}/
  data/processed/{train,dev,test}/
Then runs Whisper encoder preprocessing on each split.
"""

import argparse
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.load_l2arctic import (
    load_test_utterances,
    load_train_dev_utterances,
)


def build_split_dir_names(held_out_l1: str | None) -> dict[str, str]:
    ho_suf = "" if held_out_l1 is None else f"_heldout_{held_out_l1}"
    return {
        "train": f"train{ho_suf}",
        "dev": f"dev{ho_suf}",
        "test": "test",
    }


def load_unified_utterances(held_out_l1: str | None):
    train_utts, dev_utts = load_train_dev_utterances(held_out_l1=held_out_l1)

    test_utts = load_test_utterances()

    return {
        "train": train_utts,
        "dev": dev_utts,
        "test": test_utts,
    }


def write_raw_split(raw_split_dir: Path, utterances: list[dict]):
    if raw_split_dir.exists() and any(raw_split_dir.iterdir()):
        print(f"Raw directory already exists and is not empty: {raw_split_dir}, skipping creation.")
        return

    print(f"Creating raw directory: {raw_split_dir}")
    raw_split_dir.mkdir(parents=True, exist_ok=True)

    transcripts = {}

    for utt in tqdm(utterances, desc=f"Copying WAVs -> {raw_split_dir.name}"):
        speaker = utt["speaker"]
        utt_id = f"{speaker}_{utt['utterance_id']}"

        speaker_dir = raw_split_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        wav_dst = speaker_dir / f"{utt_id}.wav"
        src = Path(utt["wav_path"]).resolve()

        if wav_dst.exists() or wav_dst.is_symlink():
            wav_dst.unlink()

        wav_dst.symlink_to(src)

        transcripts[utt_id] = utt.get("text", "")

    with open(raw_split_dir / "data.trans.txt", "w", encoding="utf-8") as f:
        for utt_id, text in sorted(transcripts.items()):
            f.write(f"{utt_id} {text}\n")

    print(f"✅ Raw data written: {raw_split_dir}")
    print(f"Created {len(transcripts)} WAVs + data.trans.txt")


def preprocess_split(raw_split_dir: Path, processed_split_dir: Path, whisper_model: str):
    if processed_split_dir.exists() and any(processed_split_dir.iterdir()):
        print(f"Processed directory already exists and is not empty: {processed_split_dir}, skipping preprocessing.")
        return

    processed_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {raw_split_dir.name} with Whisper encoder...")
    subprocess.run(
        [
            "python",
            "-m",
            "models.whisfusion.src.data.preprocess_audio",
            "--source_dir", str(raw_split_dir),
            "--output_dir", str(processed_split_dir),
            "--model_name", whisper_model,
        ],
        check=True,
    )

    print(f"✅ Processed .pt files: {processed_split_dir}")


def preprocess_l2arctic(
    raw_output_dir: str,
    processed_output_dir: str,
    whisper_model: str = "openai/whisper-small",
    held_out_l1: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    split_dir_names = build_split_dir_names(held_out_l1)
    split_to_utts = load_unified_utterances(held_out_l1)

    for split_name in ["train", "dev", "test"]:
        utterances = split_to_utts[split_name]

        raw_split_dir = Path(raw_output_dir) / split_dir_names[split_name]
        processed_split_dir = Path(processed_output_dir) / split_dir_names[split_name]

        write_raw_split(raw_split_dir, utterances)
        preprocess_split(raw_split_dir, processed_split_dir, whisper_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess unified L2-Arctic for Whisfusion")
    parser.add_argument("--held_out_l1", default=None, help="Hold out this L1 when creating the files")
    parser.add_argument("--raw_output_dir", type=str, default="data/raw", help="Whisfusion raw dir")
    parser.add_argument("--processed_output_dir", type=str, default="data/processed", help="Whisfusion processed dir")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")

    args = parser.parse_args()

    preprocess_l2arctic(
        raw_output_dir=args.raw_output_dir,
        processed_output_dir=args.processed_output_dir,
        whisper_model=args.whisper_model,
        held_out_l1=args.held_out_l1,
    )