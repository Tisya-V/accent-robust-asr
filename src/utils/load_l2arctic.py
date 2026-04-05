"""Utilities for loading and preprocessing the L2-ARCTIC dataset
using the processed Hugging Face dataset `KoelLabs/L2Arctic`.

This module exposes two functions:

- load_scripted() -> pandas.DataFrame
- load_spontaneous() -> pandas.DataFrame

Each returned DataFrame contains:
- text, ipa (without stress markers), ipa_wstress (original IPA with stress markers)
- speaker_code, speaker_gender, speaker_native_language
- duration_s (utterance duration in seconds)
- text_num_chars, text_num_words, ipa_num_chars

You must be logged into Hugging Face and have accepted the dataset terms
if required.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import pandas as pd
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split

_DATASET_NAME = "KoelLabs/L2Arctic"


def _wav_duration_from_bytes(b: bytes) -> float:
    """Return duration in seconds from a WAV byte string."""
    with wave.open(io.BytesIO(b), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def _add_duration(batch):
    """Datasets map() helper: add a duration_s field based on audio bytes."""
    durations = []
    for audio in batch["audio"]:
        b = audio.get("bytes")
        if b is not None:
            durations.append(_wav_duration_from_bytes(b))
        else:
            durations.append(None)
    batch["duration_s"] = durations
    return batch


def _to_pandas_without_audio(hf_split) -> pd.DataFrame:
    """Convert a HF Dataset split to pandas, dropping the heavy audio column."""
    return hf_split.remove_columns(["audio"]).to_pandas()


def _add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic text/IPA length statistics to a DataFrame."""
    df = df.copy()
    df["text_num_chars"] = df["text"].str.len()
    df["text_num_words"] = df["text"].str.split().str.len()
    df["ipa_num_chars"] = df["ipa"].str.len()
    return df


def _load_split(split_name: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Internal helper to load, clean and format a single split.

    Parameters
    ----------
    split_name:
        One of "scripted" or "spontaneous".
    cache_dir:
        Optional cache directory for Hugging Face datasets.
    """
    dataset = load_dataset(_DATASET_NAME, cache_dir=cache_dir)

    # Keep audio as bytes to avoid decoding full waveforms
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Add duration in seconds
    dataset = dataset.map(_add_duration, batched=True)

    # Convert to pandas
    hf_split = dataset[split_name]
    df = hf_split.to_pandas()

    # Preserve IPA with stress, and create a stress-free version
    df["ipa_wstress"] = df["ipa"]
    df["ipa"] = df["ipa"].str.replace("[ˈˌ]", "", regex=True)

    # Add basic text stats
    df = _add_text_stats(df)

    return df


def load_scripted(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the scripted split as a preprocessed pandas DataFrame."""
    return _load_split("scripted", cache_dir=cache_dir)


def load_spontaneous(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the spontaneous split as a preprocessed pandas DataFrame."""
    return _load_split("spontaneous", cache_dir=cache_dir)

def split_dataset(df: pd.DataFrame, test_size: float = 0.15, dev_size: float = 0.15, stratify_col: str = "speaker_native_language", random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/dev/test split by L1 or speaker."""
    min_count = df[stratify_col].value_counts().min()
    
    if min_count < 6:
        # Too small to split — return all data as test, empty train/dev
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty, df.reset_index(drop=True)
    
    
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=random_state)
    train, dev = train_test_split(train_val, test_size=dev_size/(1-test_size), stratify=train_val[stratify_col], random_state=random_state)
    return train, dev, test

if __name__ == "__main__":
    scripted_df = load_scripted()
    spontaneous_df = load_spontaneous()
    print("Scripted shape:", scripted_df.shape)
    print("Scripted columns:", scripted_df.columns.tolist())
    print("Scripted: \n", scripted_df.head())
    
    train_df, dev_df, test_df = split_dataset(scripted_df)
    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print("L1 balance in test:", test_df["speaker_native_language"].value_counts())