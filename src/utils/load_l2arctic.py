"""Utilities for loading and preprocessing the L2-ARCTIC dataset
using the processed Hugging Face dataset `KoelLabs/L2Arctic`.

Exposes:
    load_scripted()       -> pd.DataFrame  (HF, for fine-tuning / WER eval)
    load_spontaneous()    -> pd.DataFrame  (HF)
    split_dataset()       -> (train, dev, test) DataFrames
                             test  = HELD_OUT_SPEAKERS (by speaker, from config)
                             train/dev = remaining speakers, 85/15 stratified by L1
    load_probe_utterances() -> list[dict]  (local files, for probing)
                               returns test-speaker utterances with wav + textgrid paths
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split

from src.config import DATASET_NAME, HELD_OUT_SPEAKERS, LOCAL_L2ARCTIC_DIR, SPEAKERS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wav_duration_from_bytes(b: bytes) -> float:
    with wave.open(io.BytesIO(b), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def _add_duration(batch):
    batch["duration_s"] = [
        _wav_duration_from_bytes(a["bytes"]) if a.get("bytes") else None
        for a in batch["audio"]
    ]
    return batch


def _add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_num_chars"] = df["text"].str.len()
    df["text_num_words"] = df["text"].str.split().str.len()
    df["ipa_num_chars"]  = df["ipa"].str.len()
    return df


def _load_split(split_name: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, cache_dir=cache_dir)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.map(_add_duration, batched=True)

    df = dataset[split_name].to_pandas()
    df["ipa_wstress"] = df["ipa"]
    df["ipa"] = df["ipa"].str.replace("[ˈˌ]", "", regex=True)
    df = _add_text_stats(df)
    return df


# ---------------------------------------------------------------------------
# Public API — HF loading
# ---------------------------------------------------------------------------

def load_scripted(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the scripted split (HF). Use for fine-tuning and WER evaluation."""
    return _load_split("scripted", cache_dir=cache_dir)


def load_spontaneous(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the spontaneous split (HF)."""
    return _load_split("spontaneous", cache_dir=cache_dir)


def split_dataset(
    df: pd.DataFrame,
    dev_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a loaded DataFrame into (train, dev, test).

    test  : rows whose speaker_code is in HELD_OUT_SPEAKERS (config.py)
            — one speaker per L1, never seen during training
    train : 85% of remaining rows, stratified by speaker_native_language
    dev   : 15% of remaining rows, stratified by speaker_native_language

    Parameters
    ----------
    df          : output of load_scripted() or load_spontaneous()
    dev_size    : fraction of non-test rows to use as dev (default 0.15)
    random_state: for reproducibility

    Returns
    -------
    train, dev, test — three DataFrames, reset index
    """
    test = df[df["speaker_code"].isin(HELD_OUT_SPEAKERS)].copy().reset_index(drop=True)
    pool = df[~df["speaker_code"].isin(HELD_OUT_SPEAKERS)].copy()

    if len(pool) == 0:
        raise ValueError("All rows are in HELD_OUT_SPEAKERS — nothing left for train/dev.")

    # Check we have enough samples per L1 to stratify
    l1_counts = pool["speaker_native_language"].value_counts()
    can_stratify = l1_counts.min() >= 2
    stratify_col = pool["speaker_native_language"] if can_stratify else None
    if not can_stratify:
        print("[WARN] Some L1 groups too small to stratify — splitting without stratification")

    train, dev = train_test_split(
        pool,
        test_size=dev_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    print(f"Split summary:")
    print(f"  Train : {len(train):>5} rows  ({len(train)/len(df)*100:.1f}%)")
    print(f"  Dev   : {len(dev):>5} rows  ({len(dev)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test):>5} rows  ({len(test)/len(df)*100:.1f}%)  [{sorted(HELD_OUT_SPEAKERS)}]")

    return train.reset_index(drop=True), dev.reset_index(drop=True), test


# ---------------------------------------------------------------------------
# Public API — local files for probing
# ---------------------------------------------------------------------------

def load_probe_utterances(
    local_root: Path = LOCAL_L2ARCTIC_DIR,
    split: str = "scripted",
    speakers: Optional[set[str]] = None,
    max_utts: Optional[int] = None,
    max_utts_per_speaker: Optional[int] = None,
) -> list[dict]:
    """
    Walk the local L2-ARCTIC directory and return a list of utterance dicts
    ready for probe_utils.build_embedding_dataset().

    Each dict contains:
        audio        : np.ndarray  (float32, 16 kHz mono)
        textgrid     : str         path to .TextGrid file
        speaker      : str         speaker code
        utterance_id : str         e.g. "arctic_a0001"

    Parameters
    ----------
    local_root : path to root of local L2-ARCTIC corpus
                 (expects speaker_dir/wav/*.wav and speaker_dir/textgrid/*.TextGrid)
    split      : "scripted" | "spontaneous" | "all"
    speakers   : if given, only load these speaker codes
                 defaults to HELD_OUT_SPEAKERS (test set) so probing is
                 always on held-out data unless you explicitly override
    max_utts   : cap total utterances loaded (useful for quick debugging)
    max_utts_per_speaker : cap utterances per speaker 
    you should set either max_utts or max_utts_per_speaker, not both
    """
    if speakers is None:
        speakers = HELD_OUT_SPEAKERS
        print(f"[INFO] load_probe_utterances: defaulting to held-out test speakers: {sorted(speakers)}")

    if speakers == {"all"}:
        speakers = SPEAKERS

    root       = Path(local_root)
    utterances = []

    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        if spk not in speakers:
            continue

        wav_dir = spk_dir / "wav"
        tg_dir  = spk_dir / "textgrid"
        if not wav_dir.exists() or not tg_dir.exists():
            print(f"  [WARN] Missing wav/ or textgrid/ for speaker {spk}, skipping")
            continue

        wav_files = sorted(wav_dir.glob("*.wav"))
        if max_utts_per_speaker:
            wav_files = wav_files[:max_utts_per_speaker]
        for wav_path in tqdm(wav_files, desc=spk, leave=False):
            uid = wav_path.stem

            if split == "scripted"    and not uid.startswith("arctic_"): continue
            if split == "spontaneous" and     uid.startswith("arctic_"): continue

            tg_path = tg_dir / (uid + ".TextGrid")
            if not tg_path.exists():
                continue

            text_path = spk_dir / "transcript" / f"{uid}.txt"
            text = text_path.read_text().strip() if text_path.exists() else None

            utterances.append({
                "wav_path":     str(wav_path),
                "text":         text,
                "textgrid":     str(tg_path),
                "speaker":      spk,
                "utterance_id": uid,
            })

            if max_utts and len(utterances) >= max_utts:
                print(f"[INFO] Reached max_utts={max_utts}, stopping early")
                return utterances

    print(f"[INFO] Loaded {len(utterances)} utterances from {len(speakers)} speakers (local)")
    return utterances


# ---------------------------------------------------------------------------
# For testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HF dataset ===")
    scripted_df = load_scripted()
    print("Scripted shape:  ", scripted_df.shape)
    print("Scripted columns:", scripted_df.columns.tolist())
    print(scripted_df[["speaker_code", "speaker_native_language", "text"]].head())

    train_df, dev_df, test_df = split_dataset(scripted_df)
    print("\nL1 balance — test:")
    print(test_df["speaker_native_language"].value_counts())
    print("\nL1 balance — train:")
    print(train_df["speaker_native_language"].value_counts())

    print("\n=== Local probe utterances (test speakers) ===")
    probe_utts = load_probe_utterances(
        local_root=LOCAL_L2ARCTIC_DIR,
        split="scripted",
        max_utts=20,
    )
    if probe_utts:
        u = probe_utts[0]
        print(f"  Sample: speaker={u['speaker']}  text={u['text']} uid={u['utterance_id']}  "
              f"wav_path={u['wav_path']}  tg={u['textgrid']}")
