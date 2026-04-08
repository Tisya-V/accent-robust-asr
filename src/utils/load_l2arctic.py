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

from sklearn.model_selection import train_test_split

from src.config import (
    LOCAL_L2ARCTIC_DIR,
    SPEAKER_L1,
    TEST_SPEAKERS,
    TRAIN_SPEAKERS,
    RANDOM_SEED,
    SUITCASE_SUBDIR,
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
        if not tg_path.exists():
            continue
        utts.append({
            "utterance_id": stem,
            "speaker":      speaker,
            "l1":           SPEAKER_L1.get(speaker, "Unknown"),
            "wav_path":     str(wav_path),
            "textgrid":     str(tg_path),
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
        speaker = spk_dir.name,
        split   = split,
        domain  = "scripted",
    )


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
    split: str = "scripted",
) -> List[Dict]:
    if split == "spontaneous":
        return load_suitcase_corpus(local_root)
    return _load_raw_scripted(Path(local_root), TEST_SPEAKERS, "test")


def load_train_dev_utterances(
    local_root:   str | Path = LOCAL_L2ARCTIC_DIR,
    dev_fraction: float      = 0.15,
    random_seed:  int        = RANDOM_SEED,
) -> tuple[List[Dict], List[Dict]]:
    """
    Utterances from the 18 non-held-out speakers split into train / dev
    at the utterance level, stratified by L1.

    Returns (train_utts, dev_utts).
    """
    utts = _load_raw_scripted(Path(local_root), TRAIN_SPEAKERS, split="train")

    train, dev = train_test_split(
        utts,
        test_size    = dev_fraction,
        random_state = random_seed,
        stratify     = [u["l1"] for u in utts],
    )
    for u in dev:
        u["split"] = "dev"

    print(f"[load_train_dev_utterances]  train={len(train)}  dev={len(dev)}  "
          f"(18 speakers, stratified by L1)")
    return list(train), list(dev)


def load_all_scripted(
    local_root:   str | Path = LOCAL_L2ARCTIC_DIR,
    dev_fraction: float      = 0.15,
    random_seed:  int        = RANDOM_SEED,
) -> List[Dict]:
    """All 24 speakers (scripted) with split labels. Used by probe scripts."""
    train, dev = load_train_dev_utterances(local_root, dev_fraction, random_seed)
    test        = load_test_utterances(local_root)
    return train + dev + test


# ---------------------------------------------------------------------------
# Public loader — spontaneous / suitcase corpus (OOD eval only)
# ---------------------------------------------------------------------------

def load_suitcase_corpus(
    local_root: str | Path = LOCAL_L2ARCTIC_DIR,
    speakers:   Set[str]   = frozenset({"all"}),
) -> List[Dict]:
    """
    Load the suitcase corpus (spontaneous speech) as an OOD evaluation set.

    All utterances are labelled split="ood", domain="spontaneous".
    Speaker identity is inferred from the filename prefix (e.g. "ABA_...").
    This data is NEVER used for training.

    Parameters
    ----------
    local_root : root of local L2-ARCTIC corpus
    speakers   : {"all"} → all speakers found; or an explicit set of IDs to keep
    """
    from src.config import SPEAKERS as ALL_SPEAKERS

    target      = set(ALL_SPEAKERS) if "all" in speakers else set(speakers)
    sc_dir      = Path(local_root) / SUITCASE_SUBDIR
    wav_dir     = sc_dir / "wav"
    tg_dir      = sc_dir / "annotation"
    txt_dir     = sc_dir / "transcript"

    if not sc_dir.exists():
        raise FileNotFoundError(
            f"Suitcase corpus not found at {sc_dir}. "
            "Check LOCAL_L2ARCTIC_DIR in config.py."
        )

    utts = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        stem = wav_path.stem
        # Infer speaker from filename prefix (e.g. "ABA_sc001" → "ABA")
        speaker = stem.upper()
        if speaker is None or speaker not in target:
            continue
        tg_path = tg_dir / f"{stem}.TextGrid"
        if not tg_path.exists():
            continue
        utts.append({
            "utterance_id": stem,
            "speaker":      speaker,
            "l1":           SPEAKER_L1.get(speaker, "Unknown"),
            "wav_path":     str(wav_path),
            "textgrid":     str(tg_path),
            "text":         _read_transcript(txt_dir / f"{stem}.txt"),
            "split":        "ood",
            "domain":       "spontaneous",
        })

    n_spk = len({u["speaker"] for u in utts})
    print(f"[load_suitcase_corpus]  {len(utts)} utterances from {n_spk} speakers")
    return utts


# ---------------------------------------------------------------------------
# Probe loader — scripted only; use load_suitcase_corpus() separately for OOD
# ---------------------------------------------------------------------------

def load_probe_utterances(
    local_root:           str | Path = LOCAL_L2ARCTIC_DIR,
    max_utts_per_speaker: int        = 50,
    speakers:             Set[str]   = frozenset({"all"}),
) -> List[Dict]:
    """
    Load scripted utterances for probing and clustering.
    For OOD / spontaneous probing, call load_suitcase_corpus() directly.

    Parameters
    ----------
    local_root           : root of local L2-ARCTIC corpus
    max_utts_per_speaker : deterministic stride-based subsample per speaker
    speakers             : {"all"} → all 24 speakers; or an explicit set of IDs
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
    # Load utterances
    # print columns and first few rows for sanity check
    train, test = load_train_dev_utterances()
    import pandas as pd
    df = pd.DataFrame(train)
    print("Columns:", df.columns.tolist())
    print(df[["speaker", "l1", "text"]].head())

    # Load suitcase corpus (OOD)
    sc_utts = load_suitcase_corpus()
    sc_df = pd.DataFrame(sc_utts)
    print("\nSuitcase corpus (OOD) columns:", sc_df.columns.tolist())
    print(sc_df[["speaker", "l1", "text"]].head())
    