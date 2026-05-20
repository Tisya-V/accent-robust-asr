#!/usr/bin/env python3
"""
Build src/analysis/cache/utterance_mapping.json with env-agnostic paths.

Paths stored as "{speaker}/{file}.pt" relative to the split root, with a
"split" field. Callers reconstruct full paths as:
    Path(os.environ["TEST_DATA_DIR"]) / info["path"]

Requires TRAIN_DATA_DIR, DEV_DATA_DIR, TEST_DATA_DIR to be set (source your
env script first, e.g. `source scripts/env.sh`).

Usage:
    source scripts/env.sh
    python src/analysis/build_utterance_mapping.py
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_SPEAKERS, SPEAKER_L1
from src.utils.textgrid import parse_textgrid, parse_lab_file


def _require_env(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        sys.exit(f"[build_utterance_mapping] {var} is not set — source your env script first.")
    return Path(val)


def build_mapping(split_dirs: dict[str, Path], l2arctic_dir: Path, cmu_arctic_dir: Path) -> dict:
    selected = {SPEAKER_L1[spk]: spk for spk in TEST_SPEAKERS}
    mapping: dict = {}
    missing = []

    for split, split_dir in split_dirs.items():
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue

        speaker_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for spk_dir in tqdm(speaker_dirs, desc=f"  {split}"):
            speaker = spk_dir.name
            l1 = next((l for l, s in selected.items() if s == speaker), None)
            if l1 is None:
                continue

            for pt_file in spk_dir.glob("*.pt"):
                prompt_id = "_".join(pt_file.stem.split("_")[-2:])

                if l1 == "English":
                    tg_candidate = cmu_arctic_dir / f"cmu_us_{speaker.lower()}_arctic" / "lab" / f"{prompt_id}.lab"
                else:
                    tg_candidate = l2arctic_dir / speaker / "textgrid" / f"{prompt_id}.TextGrid"

                speech_end_frame = None
                if tg_candidate.exists():
                    try:
                        segs = parse_textgrid(str(tg_candidate), tier_name="phones") \
                               if tg_candidate.suffix == ".TextGrid" else parse_lab_file(str(tg_candidate))
                        if segs:
                            speech_end_frame = int(segs[-1].end * 50) + 1
                    except Exception as e:
                        missing.append((prompt_id, speaker, str(e)))
                else:
                    missing.append((prompt_id, speaker, "tg/lab not found"))

                mapping.setdefault(prompt_id, {})[l1] = {
                    "speaker": speaker,
                    "split": split,
                    "path": f"{speaker}/{pt_file.name}",  # relative to split root
                    "speech_end_frame": speech_end_frame,
                }

    print(f"\n✓ {len(mapping)} prompts  |  {len(missing)} missing TextGrids/LABs")
    return mapping


if __name__ == "__main__":
    split_dirs = {
        "train": _require_env("TRAIN_DATA_DIR"),
        "dev":   _require_env("DEV_DATA_DIR"),
        "test":  _require_env("TEST_DATA_DIR"),
    }
    l2arctic_dir   = _require_env("L2ARCTIC_DIR")
    cmu_arctic_dir = _require_env("CMU_ARCTIC_DIR")

    for split, d in split_dirs.items():
        print(f"  {split:5s}: {d}")
    print(f"  l2arctic:   {l2arctic_dir}")
    print(f"  cmu_arctic: {cmu_arctic_dir}")

    mapping = build_mapping(split_dirs, l2arctic_dir, cmu_arctic_dir)

    out_path = PROJECT_ROOT / "src/analysis/cache/utterance_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Saved to {out_path}")
