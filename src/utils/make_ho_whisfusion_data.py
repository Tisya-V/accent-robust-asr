#!/usr/bin/env python3
"""
Create held-out-L1 symlinked raw + processed folders for Whisfusion.

Example:
    python make_heldout_symlinks.py --held_out_l1 Chinese

Creates:
    data/raw/train_ho_chinese/
    data/raw/dev_ho_chinese/
    data/processed/train_ho_chinese/
    data/processed/dev_ho_chinese/

by mirroring:
    data/raw/train/
    data/raw/dev/
    data/processed/train/
    data/processed/dev/

excluding speakers whose L1 matches --held_out_l1.
Files are symlinked, not copied.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

from src.utils.load_l2arctic import load_train_dev_utterances
from src.config import SPEAKER_L1, TRAIN_SPEAKERS

def normalize_l1(s: str) -> str:
    return s.strip().lower().replace(" ", "_")



def get_excluded_speakers(held_out_l1: str) -> set[str]:
    return {spk for spk in TRAIN_SPEAKERS if SPEAKER_L1[spk] == held_out_l1}



def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    dst.symlink_to(src.resolve())


def reset_dir(path: Path):
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rewrite_transcripts(src_trans: Path, dst_trans: Path, excluded_speakers: set[str]) -> int:
    kept = 0
    dst_trans.parent.mkdir(parents=True, exist_ok=True)

    with open(src_trans, "r", encoding="utf-8") as fin, open(dst_trans, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue

            utt_id = line.split(" ", 1)[0]
            speaker = utt_id.split("_", 1)[0]

            if speaker in excluded_speakers:
                continue

            fout.write(line + "\n")
            kept += 1

    return kept


def mirror_tree_with_symlinks(
    src_root: Path,
    dst_root: Path,
    excluded_speakers: set[str],
    rewrite_trans: bool = False,
) -> dict:
    if not src_root.exists():
        raise FileNotFoundError(f"Source split not found: {src_root}")

    reset_dir(dst_root)

    linked_files = 0
    kept_speakers = 0
    kept_transcripts = None

    dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    for speaker_dir in tqdm(dirs, desc="Processing speakers"):
        speaker = speaker_dir.name
        if speaker in excluded_speakers:
            continue

        kept_speakers += 1

        for src_file in sorted(speaker_dir.rglob("*")):
            if src_file.is_dir():
                continue

            rel = src_file.relative_to(src_root)
            dst_file = dst_root / rel
            safe_symlink(src_file, dst_file)
            linked_files += 1

    if rewrite_trans:
        src_trans = src_root / "data.trans.txt"
        if src_trans.exists():
            dst_trans = dst_root / "data.trans.txt"
            kept_transcripts = rewrite_transcripts(src_trans, dst_trans, excluded_speakers)

    return {
        "linked_files": linked_files,
        "kept_speakers": kept_speakers,
        "kept_transcripts": kept_transcripts,
    }


def main():
    parser = argparse.ArgumentParser(description="Create held-out-L1 symlinked raw + processed folders")
    parser.add_argument("--held_out_l1", required=True, help="L1 to exclude, e.g. Chinese")
    parser.add_argument("--raw_root", default="data/raw", help="Root raw directory")
    parser.add_argument("--processed_root", default="data/processed", help="Root processed directory")
    parser.add_argument("--src_train", default="train", help="Source train dir name")
    parser.add_argument("--src_dev", default="dev", help="Source dev dir name")
    args = parser.parse_args()

    suffix = normalize_l1(args.held_out_l1)
    excluded_speakers = get_excluded_speakers(args.held_out_l1)

    print(f"Held-out L1: {args.held_out_l1}")
    print(f"Excluded speakers ({len(excluded_speakers)}): {sorted(excluded_speakers)}")

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)

    jobs = [
        {
            "src": raw_root / args.src_train,
            "dst": raw_root / f"train_ho_{suffix}",
            "rewrite_trans": True,
            "label": "raw/train",
        },
        {
            "src": raw_root / args.src_dev,
            "dst": raw_root / f"dev_ho_{suffix}",
            "rewrite_trans": True,
            "label": "raw/dev",
        },
        {
            "src": processed_root / args.src_train,
            "dst": processed_root / f"train_ho_{suffix}",
            "rewrite_trans": False,
            "label": "processed/train",
        },
        {
            "src": processed_root / args.src_dev,
            "dst": processed_root / f"dev_ho_{suffix}",
            "rewrite_trans": False,
            "label": "processed/dev",
        },
    ]

    for job in jobs:
        print(f"\nCreating {job['label']}:")
        stats = mirror_tree_with_symlinks(
            src_root=job["src"],
            dst_root=job["dst"],
            excluded_speakers=excluded_speakers,
            rewrite_trans=job["rewrite_trans"],
        )
        print(f"  -> {job['dst']}")
        print(f"  stats: {stats}")


if __name__ == "__main__":
    main()