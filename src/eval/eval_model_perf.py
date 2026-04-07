"""
eval_model_perf.py
Inference + WER/CER computation for baseline comparison.
Designed to run as a SLURM job — all compute-heavy work happens here,
results cached to CSV for later notebook visualisation.

Usage:
    python eval_model_perf.py --models baseline,baseline_lora --splits scripted,spontaneous
    python eval_model_perf.py --models ctc_aux --splits scripted

    # Or via SLURM:
    sbatch --gres=gpu:1 --wrap="python eval_model_perf.py --models baseline,baseline_lora"
"""

import argparse
import os
import re
import torch
import pandas as pd
from pathlib import Path
from jiwer import wer as jiwer_wer, cer as jiwer_cer

from src.utils.audio_utils import bytes_to_array
from src.utils.load_l2arctic import load_scripted, load_spontaneous, split_dataset
from src.utils.model_loader import get_model_registry
from src.config import LOCAL_L2ARCTIC_DIR

BASELINE_ID = "openai/whisper-small"
BATCH_SIZE  = 8


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def norm(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def add_normalized_columns(df, ref_col="text", pred_col="prediction"):
    df = df.copy()
    df["reference_norm"]  = df[ref_col].apply(norm)
    df["prediction_norm"] = df[pred_col].apply(norm)
    return df


def attach_utterance_stats(df):
    df = df.copy()
    df["ref_num_words"] = df["reference_norm"].str.split().str.len()
    df["ref_num_chars"] = df["reference_norm"].str.len()
    return df


def attach_utt_wer(df, ref_col="reference_norm", pred_col="prediction_norm"):
    df = df.copy()
    df["utt_wer"] = df.apply(
        lambda r: jiwer_wer(r[ref_col], r[pred_col]) if r[ref_col] else None, axis=1
    )
    df["utt_cer"] = df.apply(
        lambda r: jiwer_cer(r[ref_col], r[pred_col]) if r[ref_col] else None, axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def transcribe(df, processor, model, device, batch_size=BATCH_SIZE):
    predictions = []
    for start in range(0, len(df), batch_size):
        batch_df     = df.iloc[start:start + batch_size]
        audio_arrays = [bytes_to_array(row["audio"]["bytes"]) for _, row in batch_df.iterrows()]
        inputs       = processor(
            audio_arrays, sampling_rate=16000,
            return_tensors="pt", truncation=True, return_attention_mask=True,
        )
        with torch.no_grad():
            pred_ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                language="en", task="transcribe",
            )
        predictions.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
        if (start // batch_size) % 10 == 0:
            print(f"  {start}/{len(df)}", end="\r")
    print(f"  {len(df)}/{len(df)} done")
    return predictions


def build_results(df, predictions):
    results = df.drop(columns=["audio"]).copy()
    results["prediction"] = predictions
    results = add_normalized_columns(results)
    results = attach_utterance_stats(results)
    results = attach_utt_wer(results)
    return results


# ---------------------------------------------------------------------------
# Per-split, per-model runner
# ---------------------------------------------------------------------------

def run_one(model_key, split, df, registry, device, output_dir):
    out_path = Path(output_dir) / f"{model_key}_{split}_predictions.csv"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return

    print(f"  Loading model [{model_key}] …")
    model, processor = registry[model_key]["loader"]()
    model.eval()
    model.generation_config.suppress_tokens       = None
    model.generation_config.begin_suppress_tokens = None

    print(f"  Transcribing {len(df)} utterances [{split}] …")
    preds   = transcribe(df, processor, model, device)
    results = build_results(df, preds)
    results.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",      default="baseline",
                        help=f"Comma-separated model keys, options: {get_model_registry(device).keys()}")
    parser.add_argument("--splits",      default="scripted,spontaneous",
                        help="Comma-separated splits: scripted, spontaneous")
    parser.add_argument("--output_dir",  default="results/model_perf_comparison")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"=== eval_model_perf === device={device}")

    model_keys = [k.strip() for k in args.models.split(",")]
    splits     = [s.strip() for s in args.splits.split(",")]
    registry   = get_model_registry(device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets once
    datasets = {}
    if "scripted" in splits:
        scripted_df = load_scripted()
        _, _, test  = split_dataset(scripted_df)
        datasets["scripted"] = test.reset_index(drop=True)
        print(f"Scripted test : {len(datasets['scripted'])} utterances")
    if "spontaneous" in splits:
        datasets["spontaneous"] = load_spontaneous().reset_index(drop=True)
        print(f"Spontaneous   : {len(datasets['spontaneous'])} utterances")

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key: '{key}'. Available: {list(registry.keys())}")
        print(f"\n[Model: {key}]")
        for split, df in datasets.items():
            run_one(key, split, df, registry, device, args.output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
