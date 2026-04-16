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
    cmu_root: str | Path,
    speakers: Set[str] = frozenset({"bdl"}),
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
                "split": "test",
                "domain": "scripted",
            })

        if max_utts_per_speaker and len(spk_utts) > max_utts_per_speaker:
            step = max(1, len(spk_utts) // max_utts_per_speaker)
            spk_utts = spk_utts[::step][:max_utts_per_speaker]

        utts.extend(spk_utts)

    print(f"[load_cmu_arctic_utterances] {len(utts)} utterances from {len(speakers)} speakers")
    return utts

def _load_edacc_utterances(
    manifest_path: str | Path,
    max_utts: int | None = None,
) -> List[Dict]:
    """
    Load the pre-cut Jamaican EdAcc subset from a CSV manifest.
    Expect columns:
      utteranceid,speaker,l1,accent,text,wavpath,split,corpus,recording_id,start,end,duration,n_words
    """
    import pandas as pd

    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)

    if max_utts is not None:
        df = df.head(max_utts).copy()

    utts = []
    for _, row in df.iterrows():
        utts.append(
            {
                "utterance_id": str(row["utteranceid"]),
                "speaker": str(row["speaker"]),
                "l1": "Jamaican",
                "wav_path": str(row["wavpath"]),
                "textgrid": None,
                "text": str(row.get("text", "")),
                "split": "test",
                "domain": "spontaneous",
            }
        )
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
    split: str = "scripted",
    include_cmu_native: bool = True,
    cmu_root: str | Path | None = "data",
    cmu_speakers: Set[str] = frozenset({"bdl"}),
    max_cmu_utts_per_speaker: int | None = None,
    include_edacc_jamaican: bool = True,
    edacc_manifest_path: str | Path | None  = "data/edacc_jamaican_subset/jamaican_subset_all.csv",
    max_edacc_utts: int | None = None,
) -> List[Dict]:
    if split == "spontaneous":
        utts =  load_suitcase_corpus(local_root)
    
        if include_edacc_jamaican:
            if edacc_manifest_path is None:
                raise ValueError("include_edacc_jamaican=True but edacc_manifest_path was not provided")
            utts.extend(
                _load_edacc_utterances(
                    manifest_path=edacc_manifest_path,
                    max_utts=max_edacc_utts,
                )
            )

        return utts

    utts = _load_raw_scripted(Path(local_root), TEST_SPEAKERS, "test")

    if include_cmu_native:
        if cmu_root is None:
            raise ValueError("include_cmu_native=True but cmu_root was not provided")
        utts.extend(
            _load_cmu_arctic_utterances(
                cmu_root=cmu_root,
                speakers=cmu_speakers,
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
) -> tuple[List[Dict], List[Dict]]:
    """
    Utterances from the 18 non-held-out speakers split into train / dev
    at the utterance level, stratified by L1.

    Returns (train_utts, dev_utts).
    """

    speakers = set(TRAIN_SPEAKERS)
    if held_out_l1:
        speakers = {spk for spk in TRAIN_SPEAKERS if SPEAKER_L1.get(spk) != held_out_l1}

    utts = _load_raw_scripted(Path(local_root), speakers, split="train")

    train, dev = train_test_split(
        utts,
        test_size    = dev_fraction,
        random_state = random_seed,
        stratify     = [u["l1"] for u in utts],
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
    # # Load utterances
    # # print columns and first few rows for sanity check
    # train, dev = load_train_dev_utterances()
    import pandas as pd
    # df = pd.DataFrame(train)
    # print("Columns:", df.columns.tolist())
    # print(df[["speaker", "l1", "text"]].head())


    # Load test utterances
    import numpy as np
    test = load_test_utterances(max_cmu_utts_per_speaker=10, split="spontaneous")
    test_df = pd.DataFrame(test)
    test_df = test_df[test_df["l1"] == "Jamaican" ]
    print(test_df.head(15))

    # # Load suitcase corpus (OOD)
    # sc_utts = load_suitcase_corpus()
    # sc_df = pd.DataFrame(sc_utts)
    # print("\nSuitcase corpus (OOD) columns:", sc_df.columns.tolist())
    # print(sc_df[["speaker", "l1", "text"]].head())

    