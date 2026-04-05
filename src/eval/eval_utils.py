from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
from jiwer import wer, cer


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def add_normalized_columns(
    df: pd.DataFrame,
    ref_col: str = "text",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    df = df.copy()
    df["reference_norm"] = df[ref_col].fillna("").map(normalize_text)
    df["prediction_norm"] = df[pred_col].fillna("").map(normalize_text)
    return df


def compute_metrics(
    refs: List[str],
    preds: List[str],
) -> Dict[str, float]:
    return {
        "wer": wer(refs, preds),
        "cer": cer(refs, preds),
    }


def compute_metrics_df(
    df: pd.DataFrame,
    ref_col: str = "reference_norm",
    pred_col: str = "prediction_norm",
) -> Dict[str, float]:
    refs = df[ref_col].tolist()
    preds = df[pred_col].tolist()
    return compute_metrics(refs, preds)


def compute_grouped_metrics(
    df: pd.DataFrame,
    group_col: str = "speaker_native_language",
    ref_col: str = "reference_norm",
    pred_col: str = "prediction_norm",
) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col):
        m = compute_metrics_df(group_df, ref_col=ref_col, pred_col=pred_col)
        rows.append({
            group_col: group_value,
            "num_utts": len(group_df),
            "wer": m["wer"],
            "cer": m["cer"],
        })
    return pd.DataFrame(rows).sort_values("wer", ascending=True).reset_index(drop=True)


def attach_utterance_stats(
    df: pd.DataFrame,
    ref_col: str = "reference_norm",
    pred_col: str = "prediction_norm",
) -> pd.DataFrame:
    df = df.copy()
    df["ref_num_words"] = df[ref_col].str.split().str.len()
    df["pred_num_words"] = df[pred_col].str.split().str.len()
    df["exact_match"] = df[ref_col] == df[pred_col]
    return df