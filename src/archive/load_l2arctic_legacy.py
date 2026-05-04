"""
src/archive/load_l2arctic_legacy.py
Legacy data loaders for EdAcc and spontaneous/suitcase corpus (no longer used in main pipeline).

Kept for reference in case these datasets are needed again.
Use src/utils/load_l2arctic.py for the current main pipeline.
"""

from pathlib import Path
from typing import Dict, List, Set

from src.config import SPEAKER_L1, SPONTANEOUS_SUBDIR, SUITCASE_SUBDIR, LOCAL_L2ARCTIC_DIR


def _read_transcript(path: Path) -> str:
    """Helper to read transcript file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""


def _load_edacc_utterances(
    manifest_path: str | Path,
    l1: str | None = None,
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
                "l1": l1,
                "wav_path": str(row["wavpath"]),
                "textgrid": None,
                "text": str(row.get("text", "")),
                "split": "test",
                "domain": "spontaneous",
            }
        )
    return utts


def load_spontaneous(path):
    """
    Load spontaneous utterances from a preprocessed CSV manifest.
    Assumes preprocessing via preprocess_l2arctic_spontaneous.py.
    """
    import pandas as pd
    df = pd.read_csv(path)

    return [
        {
            "utterance_id": r["utterance_id"],
            "speaker": r["speaker"],
            "l1": r["l1"],
            "wav_path": r["wav_path"],
            "textgrid": None,
            "text": r["text"],
            "split": "ood",
            "domain": "spontaneous",
        }
        for _, r in df.iterrows()
    ]


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
