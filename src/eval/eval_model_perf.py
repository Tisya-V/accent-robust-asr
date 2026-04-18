"""
eval_model_perf.py
Inference + WER/PER computation across models and splits.
Designed to run as a SLURM job — results cached to CSV for notebook visualisation.

Scripted     → 6 held-out test speakers, never seen during training.
Spontaneous  → suitcase corpus (OOD, all speakers).

WER: word error rate on normalised text.
PER: phoneme error rate — G2P(prediction) vs G2P(reference), both via g2p_en.
     (text-derived, labels as "PER (G2P)" in reporting)

Usage:
    python eval_model_perf.py --models baseline,baseline_lora --splits scripted spontaneous
    python eval_model_perf.py --models ctc_aux --splits scripted

    # Via SLURM:
    sbatch --gres=gpu:1 --wrap="python eval_model_perf.py --models baseline,baseline_lora"

Output:
    results/model_perf_comparison/{model_key}_{split}_predictions.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from jiwer import wer as jiwer_wer
from tqdm import tqdm

from src.config import LOCAL_L2ARCTIC_DIR, NLTK_DATA_PATH
from src.utils.load_l2arctic import load_test_utterances
from src.utils.model_loader import get_model_registry

import os
import nltk
nltk.data.path.insert(0, NLTK_DATA_PATH)
os.environ["NLTK_DATA"] = NLTK_DATA_PATH

import g2p_en

BATCH_SIZE = 8

_G2P = g2p_en.G2p()

def text_to_phones(text: str) -> list[str]:
    """Normalised text → ARPAbet phone list (stress digits stripped)."""
    raw = _G2P(text)
    return [p.rstrip("012") for p in raw if p.strip() and p[0].isalpha()]


def utt_per(ref: str, pred: str) -> float | None:
    """G2P-derived phoneme error rate for one utterance."""
    if not ref:
        return None
    ref_phones  = " ".join(text_to_phones(ref))
    pred_phones = " ".join(text_to_phones(pred))
    if not ref_phones:
        return None
    return float(jiwer_wer(ref_phones, pred_phones))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+",     " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def transcribe(
    utterances: list[dict],
    processor,
    model,
    device:     str,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    import librosa
    import soundfile as sf

    predictions = []
    pbar = tqdm(range(0, len(utterances), batch_size),
                desc="  transcribing", unit="batch", dynamic_ncols=True)
    for start in pbar:
        batch  = utterances[start : start + batch_size]
        audios = []
        for utt in batch:
            audio, sr = sf.read(utt["wav_path"], dtype="float32", always_2d=False)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            if utt.get("domain") == "spontaneous":
                predictions.append(transcribe_long(audio, processor, model, device)) #reroute
                continue
            else:
                audios.append(audio)  # handle scripted as before in batches

        if not audios:
            continue
        inputs = processor(
            audios, sampling_rate=16000,
            return_tensors="pt", truncation=True, return_attention_mask=True,
        )
        with torch.no_grad():
            pred_ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                language="en", 
                task="transcribe",
                # no_repeat_ngram_size=3,
                # repetition_penalty=1.1,
                temperature=0.0,
                
            )
        predictions.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
    return predictions

def transcribe_long(audio, processor, model, device, chunk_s=28):
    # note that this has hard cutoffs
    # so may split words and stuff
    # but this would be like max 4-8 words in a very long 1/2 min utterance
    # so mostly hidden but worth noting
    sr       = 16000
    chunk_len = chunk_s * sr

    if len(audio) <= chunk_len:
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                language="en", task="transcribe", temperature=0.0,
            )
        return processor.decode(ids[0], skip_special_tokens=True).strip()

    parts = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < sr:  # skip sub-1s tail
            break
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                language="en", task="transcribe", temperature=0.0,
            )
        parts.append(processor.decode(ids[0], skip_special_tokens=True).strip())

    return " ".join(parts)

# ---------------------------------------------------------------------------
# Results assembly
# ---------------------------------------------------------------------------

def build_results(utterances: list[dict], predictions: list[str]) -> pd.DataFrame:
    rows = []
    for utt, pred in zip(utterances, predictions):
        ref    = norm(utt["text"])
        pred_n = norm(pred)
        rows.append({
            "utterance_id":    utt["utterance_id"],
            "speaker":         utt["speaker"],
            "l1":              utt["l1"],
            "domain":          utt["domain"],
            "wav_path":        utt["wav_path"],
            "text":            utt["text"],
            "prediction":      pred,
            "reference_norm":  ref,
            "prediction_norm": pred_n,
            "ref_num_words":   len(ref.split()),
            "utt_wer":         float(jiwer_wer(ref, pred_n)) if ref else None,
            "utt_per":         utt_per(ref, pred_n),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-split, per-model runner
# ---------------------------------------------------------------------------

def run_one(
    model_key:  str,
    split:      str,
    utterances: list[dict],
    registry:   dict,
    device:     str,
    output_dir: str,
) -> None:
    out_path = Path(output_dir) / f"{model_key}_{split}_predictions.csv"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    print(f"  Loading model [{model_key}] ...")
    model, processor = registry[model_key]["loader"]()
    model.eval()

    print(f"  Transcribing {len(utterances):,} utterances [{split}] ...")
    preds   = transcribe(utterances, processor, model, device)
    results = build_results(utterances, preds)
    results.to_csv(out_path, index=False)

    wer = float(jiwer_wer(results["reference_norm"].tolist(),
                           results["prediction_norm"].tolist()))
    per_vals = results["utt_per"].dropna()
    per_str  = f"  PER={per_vals.mean():.3f}" if len(per_vals) else "  PER=n/a"
    print(f"  WER={wer:.3f}{per_str}  →  {out_path}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--models",     default="baseline",
                   help="Comma-separated keys from MODEL_REGISTRY")
    p.add_argument("--splits",     default=["scripted", "spontaneous"], nargs="+",
                   help="scripted | spontaneous  (space- or comma-separated)")
    p.add_argument("--output_dir", default="results/model_perf_comparison")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    splits     = [s for tok in args.splits for s in tok.split(",")]
    model_keys = [k.strip() for k in args.models.split(",")]

    print(f"=== eval_model_perf  device={device} ===")
    print(f"    models  : {model_keys}")
    print(f"    splits  : {splits}")

    registry = get_model_registry(device)
    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model \'{key}\'. Available: {sorted(registry)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[dict]] = {}
    for split in splits:
        if split not in ("scripted", "spontaneous"):
            raise ValueError(f"Unknown split \'{split}\'. Choose: scripted, spontaneous")
        utts = load_test_utterances(local_root=args.data_root, split=split)
        datasets[split] = utts
        print(f"  {split}: {len(utts):,} utterances")

    for key in model_keys:
        print(f"\\n[Model: {key}]")
        for split, utterances in datasets.items():
            run_one(key, split, utterances, registry, device, args.output_dir)

    print("\\nAll done.")


if __name__ == "__main__":
    main()