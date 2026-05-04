"""
src/utils/load_l2arctic.py
Local L2-ARCTIC data loader — single source of truth for all scripts.

Corpus layout
-------------
Scripted (main corpus):
    <LOCAL_L2ARCTIC_DIR>/
        <SPEAKER>/
            wav/          <SPEAKER>_arctic_<id>.wav
            annotation/   <SPEAKER>_arctic_<id>.TextGrid
            transcript/   <SPEAKER>_arctic_<id>.txt

Spontaneous (suitcase corpus — OOD eval only, no train/dev/test split):
    <LOCAL_L2ARCTIC_DIR>/
        suitcase_corpus/
            wav/          <SPEAKER>_<id>.wav
            annotation/   <SPEAKER>_<id>.TextGrid
            transcript/   <SPEAKER>_<id>.txt   (may be absent)

All utterance dicts have keys:
    utterance_id  str   e.g. "ABA_arctic_a0001"
    speaker       str   e.g. "ABA"
    l1            str   e.g. "Arabic"
    wav_path      str   absolute path to .wav
    textgrid      str   absolute path to .TextGrid
    text          str   transcript (empty string if missing)
    split         str   "train" | "dev" | "test" | "ood"
    domain        str   "scripted" | "spontaneous"
"""

from __future__ import annotations

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
import re

from sklearn.model_selection import train_test_split

from src.config import (
    LOCAL_L2ARCTIC_DIR,
    CMU_ARCTIC_DIR,
    SPEAKER_L1,
    TEST_SPEAKERS,
    TRAIN_SPEAKERS,
    RANDOM_SEED,
)



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_transcript(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""


def _iter_dir(
    wav_dir: Path,
    tg_dir:  Path,
    annt_dir:Path,
    txt_dir: Path,
    speaker: str,
    split:   str,
    domain:  str,
) -> List[Dict]:
    """Yield utterance dicts from a wav/ + annotation/ + transcript/ triplet."""
    if not wav_dir.exists():
        return []
    utts = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        stem    = wav_path.stem
        tg_path = tg_dir / f"{stem}.TextGrid"
        annt_path = annt_dir / f"{stem}.TextGrid"
        if not annt_path.exists():
            annt_path = None
        if not tg_path.exists():
            continue
        utts.append({
            "utterance_id": stem,
            "speaker":      speaker,
            "l1":           SPEAKER_L1.get(speaker, "Unknown"),
            "wav_path":     str(wav_path),
            "textgrid":     str(tg_path),
            "annotation":   str(annt_path),
            "text":         _read_transcript(txt_dir / f"{stem}.txt"),
            "split":        split,
            "domain":       domain,
        })
    return utts


def _iter_speaker_scripted(spk_dir: Path, split: str) -> List[Dict]:
    return _iter_dir(
        wav_dir = spk_dir / "wav",
        tg_dir  = spk_dir / "textgrid",
        txt_dir = spk_dir / "transcript",
        annt_dir = spk_dir / "annotation",
        speaker = spk_dir.name,
        split   = split,
        domain  = "scripted",
    )


def _parse_cmu_txt_done_data(path: Path) -> Dict[str, str]:
    """
    Parse CMU ARCTIC etc/txt.done.data into:
        {"arctic_a0001": "AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC", ...}

    Expected line format:
        ( arctic_a0001 "AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC" )
    """
    if not path.exists():
        raise FileNotFoundError(f"CMU ARCTIC transcript file not found: {path}")

    mapping: Dict[str, str] = {}
    pattern = re.compile(r'^\(\s*([^\s]+)\s+"(.*)"\s*\)\s*$')

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            print(f"[WARN] Could not parse txt.done.data line: {line}")
            continue
        utt_id, text = m.groups()
        mapping[utt_id] = text

    return mapping


def _load_cmu_arctic_utterances(
    cmu_root: str | Path | None = None,
    speakers: Set[str] = frozenset({"bdl"}),
    split: str = "test",
    max_utts_per_speaker: int | None = None,
) -> List[Dict]:
    """
    Load native-English CMU ARCTIC speakers into the same utterance schema
    used by L2-ARCTIC loaders.

    Expected layout:
        <cmu_root>/
            cmu_us_bdl_arctic/
                wav/
                    arctic_a0001.wav
                etc/
                    txt.done.data

    Returns utterance dicts with:
        utterance_id, speaker, l1, wav_path, textgrid, text, split, domain
    """
    if cmu_root is None:
        cmu_root = CMU_ARCTIC_DIR
    cmu_root = Path(cmu_root)
    utts: List[Dict] = []

    for speaker in sorted(speakers):
        spk_dir = cmu_root / f"cmu_us_{speaker}_arctic"
        wav_dir = spk_dir / "wav"
        txt_done = spk_dir / "etc" / "txt.done.data"

        if not wav_dir.exists():
            print(f"[WARN] CMU ARCTIC wav dir not found: {wav_dir}")
            continue
        if not txt_done.exists():
            print(f"[WARN] CMU ARCTIC transcript file not found: {txt_done}")
            continue

        transcript_map = _parse_cmu_txt_done_data(txt_done)

        spk_utts = []
        for wav_path in sorted(wav_dir.glob("*.wav")):
            stem = wav_path.stem
            text = transcript_map.get(stem, "")
            spk_utts.append({
                "utterance_id": f"{speaker}_{stem}",
                "speaker": speaker.upper(),
                "l1": "English",
                "wav_path": str(wav_path),
                "textgrid": None,
                "text": text,
                "split": split,
                "domain": "scripted",
            })

        if max_utts_per_speaker and len(spk_utts) > max_utts_per_speaker:
            step = max(1, len(spk_utts) // max_utts_per_speaker)
            spk_utts = spk_utts[::step][:max_utts_per_speaker]

        utts.extend(spk_utts)

    print(f"[_load_cmu_arctic_utterances] {len(utts)} utterances from {len(speakers)} speakers")
    return utts

def _load_raw_scripted(
    local_root: Path,
    speakers:   Set[str],
    split:      str,
) -> List[Dict]:
    utts = []
    for spk in tqdm(sorted(speakers), desc=f"Loading scripted utterances from {len(speakers)} speakers"):
        spk_dir = local_root / spk
        if not spk_dir.is_dir():
            print(f"  [WARN] Speaker dir not found: {spk_dir}")
            continue
        utts.extend(_iter_speaker_scripted(spk_dir, split))
    return utts


# ---------------------------------------------------------------------------
# Public loaders — scripted
# ---------------------------------------------------------------------------

def load_test_utterances(
    local_root: str | Path = LOCAL_L2ARCTIC_DIR,
    test_speakers: Set[str] = TEST_SPEAKERS,
    include_cmu_native: bool = True,
    cmu_root: str | Path | None = None,
    cmu_speakers: Set[str] = frozenset({"bdl"}),
    max_cmu_utts_per_speaker: int | None = None,
) -> List[Dict]:

    utts = _load_raw_scripted(Path(local_root), test_speakers, "test")

    if include_cmu_native:
        utts.extend(
            _load_cmu_arctic_utterances(
                cmu_root=cmu_root,
                speakers=cmu_speakers,
                split="test",
                max_utts_per_speaker=max_cmu_utts_per_speaker,
            )
        )

    for u in utts:
        u["split"] = "test"

    print(f"[load_test_utterances] {len(utts)} utterances total")
    return utts

def load_train_dev_utterances(
    local_root:   str | Path = LOCAL_L2ARCTIC_DIR,
    dev_fraction: float      = 0.15,
    random_seed:  int        = RANDOM_SEED,
    held_out_l1:  str | None   = None,
    include_cmu_native: bool = True,
    cmu_root: str | Path | None = None,
    cmu_speakers: Set[str] = frozenset({"clb", "rms", "slt"}),
    max_cmu_utts_per_speaker: int | None = None,
) -> tuple[List[Dict], List[Dict]]:
    """
    Utterances from non-held-out speakers split into train / dev
    at the utterance level, stratified by L1.

    Includes:
    - L2-ARCTIC train speakers (18 speakers from 6 L1s)
    - CMU-ARCTIC native English speakers (clb, rms, slt by default; bdl is test-only)

    Returns (train_utts, dev_utts).
    """

    speakers = set(TRAIN_SPEAKERS)
    if held_out_l1:
        speakers = {spk for spk in TRAIN_SPEAKERS if SPEAKER_L1.get(spk) != held_out_l1}

    utts_scripted = _load_raw_scripted(Path(local_root), speakers, split="train")

    # Add CMU train speakers (clb, rms, slt)
    if include_cmu_native:
        utts_cmu = _load_cmu_arctic_utterances(
            cmu_root=cmu_root,
            speakers=cmu_speakers,
            split="train",
            max_utts_per_speaker=max_cmu_utts_per_speaker,
        )
        utts_scripted.extend(utts_cmu)

    train, dev = train_test_split(
        utts_scripted,
        test_size    = dev_fraction,
        random_state = random_seed,
        stratify     = [u["l1"] for u in utts_scripted],
    )
    for u in dev:
        u["split"] = "dev"

    print(f"[load_train_dev_utterances]  train={len(train)}  dev={len(dev)}  "
          f"held_out_l1={held_out_l1}")

    return list(train), list(dev)


def load_all_scripted(
    local_root:   str | Path = LOCAL_L2ARCTIC_DIR,
    dev_fraction: float      = 0.15,
    random_seed:  int        = RANDOM_SEED,
) -> List[Dict]:
    """All scripted speakers (L2-ARCTIC + CMU) with split labels. Used by probe scripts."""
    train, dev = load_train_dev_utterances(local_root, dev_fraction, random_seed)
    test        = load_test_utterances(local_root)
    return train + dev + test


# ---------------------------------------------------------------------------
# Probe loader — scripted only
# ---------------------------------------------------------------------------

def load_probe_utterances(
    local_root:           str | Path = LOCAL_L2ARCTIC_DIR,
    max_utts_per_speaker: int        = 50,
    speakers:             Set[str]   = frozenset({"all"}),
) -> List[Dict]:
    """
    Load scripted utterances for probing and clustering.

    Parameters
    ----------
    local_root           : root of local L2-ARCTIC corpus
    max_utts_per_speaker : deterministic stride-based subsample per speaker
    speakers             : {"all"} → all scripted speakers; or an explicit set of IDs
    """
    from src.config import SPEAKERS as ALL_SPEAKERS

    target   = set(ALL_SPEAKERS) if "all" in speakers else set(speakers)
    all_utts = load_all_scripted(local_root)
    utts     = [u for u in all_utts if u["speaker"] in target]

    if max_utts_per_speaker:
        by_spk: Dict[str, List[Dict]] = defaultdict(list)
        for u in utts:
            by_spk[u["speaker"]].append(u)
        utts = []
        for spk, spk_utts in sorted(by_spk.items()):
            if len(spk_utts) > max_utts_per_speaker:
                step     = max(1, len(spk_utts) // max_utts_per_speaker)
                spk_utts = spk_utts[::step][:max_utts_per_speaker]
            utts.extend(spk_utts)

    print(f"[load_probe_utterances]  {len(utts)} utterances "
          f"from {len(target)} speakers")
    return utts


if __name__ == "__main__":
    # # Load utterances
    train, dev = load_train_dev_utterances(held_out_l1="Chinese")
    import pandas as pd
    train_df = pd.DataFrame(train)
    speakers = train_df["speaker"].unique()
    domains = train_df["domain"].unique()
    print(f"Train speakers [found {len(speakers)}]: {sorted(speakers)}")
    print(f"Train domains: {sorted(domains)}")
    # print("Columns:", df.columns.tolist())
    # print(df[["speaker", "l1", "text"]].head())


    # Load test utterances
    import numpy as np
    test = load_test_utterances(max_cmu_utts_per_speaker=10)
    test_df = pd.DataFrame(test)
    # print speakers
    speakers = test_df["speaker"].unique()
    domains = test_df["domain"].unique()
    print(f"Test speakers [found {len(speakers)}]: {sorted(speakers)}")
    print(f"Test domains: {sorted(domains)}")