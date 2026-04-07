"""
probe_phoneme.py
Linear probing for PHONEME IDENTITY from Whisper encoder layers.

For each encoder layer (0-6) of each model:
  - Trains a logistic regression on segment-level mean-pooled embeddings
  - Predicts ARPAbet phone class (39-way classification)
  - Evaluates accuracy, macro-F1, and per-phone F1
  - Speaker-held-out cross-validation to prevent leakage

Usage:
    python probe_phoneme.py \
        --data_root /path/to/l2arctic \
        --baseline_model openai/whisper-small \
        --lora_model   /path/to/lora-checkpoint \
        --split        scripted \
        --output_dir   results/phoneme_probe \
        [--max_utts 500]
"""

import argparse
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupKFold

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel  # only needed if lora_model is set

from src.config import LOCAL_L2ARCTIC_DIR
from src.eval.probing.probe_utils import (
    ARPABET_VOCAB, NUM_PHONES, WHISPER_N_ENCODER_LAYERS,
    build_embedding_dataset, records_to_arrays, save_results,
    SPEAKER_L1,
)

from src.utils.load_l2arctic import load_probe_utterances



# ---------------------------------------------------------------------------
# Probe training & evaluation
# ---------------------------------------------------------------------------

def run_phoneme_probe(X, y, groups, layer_idx: int, n_folds: int = 5):
    """
    Speaker-held-out logistic regression probe for phoneme identity.

    Args:
        X:          (N, D) embeddings
        y:          (N,)  phone labels (int)
        groups:     (N,)  speaker IDs for group-kfold split
        layer_idx:  just for logging

    Returns:
        dict with accuracy, macro_f1, per_phone_f1
    """
    gkf = GroupKFold(n_splits=n_folds)
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = SGDClassifier(
            max_iter=300, loss="log_loss", random_state=42,
        )
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)

        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc      = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    # Per-phone F1 — indexed by phone ID
    f1_per_phone_arr = f1_score(
        all_true, all_preds,
        labels=list(range(NUM_PHONES)),
        average=None,
        zero_division=0,
    )
    per_phone_f1 = {ARPABET_VOCAB[i]: float(f1_per_phone_arr[i])
                    for i in range(NUM_PHONES)}

    print(f"    Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={macro_f1:.3f}")
    return {
        "accuracy":      float(acc),
        "macro_f1":      float(macro_f1),
        "per_phone_f1":  per_phone_f1,
        "n_samples":     int(len(all_true)),
        "n_folds":       n_folds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phoneme linear probe on Whisper encoder")
    parser.add_argument("--data_root",       default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--baseline_model",  default="openai/whisper-small")
    parser.add_argument("--lora_model",      default=None,   help="Path to LoRA checkpoint dir")
    parser.add_argument("--split",           default="scripted",
                        choices=["scripted","spontaneous","all"])
    parser.add_argument("--output_dir",      default="results/phoneme_probe")
    parser.add_argument("--layers",          default=None,
                        help="Comma-separated layer indices, e.g. 0,1,2,3,4,5,6")
    parser.add_argument("--max_utts",        type=int, default=None)
    parser.add_argument("--n_folds",         type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Phoneme Probe ===")
    print(f"Running on device: {device}")

    layer_indices = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(WHISPER_N_ENCODER_LAYERS + 1))
    )

    # ------------------------------------------------------------------
    # Load utterances
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading utterances from {args.data_root}  (split={args.split})")
    utterances = load_probe_utterances(
        local_root=args.data_root,
        split=args.split,
        max_utts=args.max_utts,
    )
    print(f"      Found {len(utterances)} utterances")

    # ------------------------------------------------------------------
    # Models to evaluate
    # ------------------------------------------------------------------
    models_to_eval = {"baseline": args.baseline_model}
    if args.lora_model:
        models_to_eval["lora"] = args.lora_model

    all_results = {}

    for model_name, model_path in models_to_eval.items():
        print(f"\n[2/4] Loading model: {model_name}  ({model_path})")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        if model_name == "lora" and args.lora_model:
            base = PeftModel.from_pretrained(base, model_path)
            base = base.merge_and_unload()
        base = base.to(device)

        # ------------------------------------------------------------------
        # Extract embeddings at all requested layers
        # ------------------------------------------------------------------
        print(f"[3/4] Extracting encoder hidden states …")
        records = build_embedding_dataset(
            model=base,
            processor=processor,
            utterances=utterances,
            layer_indices=layer_indices,
            device=device,
        )
        print(f"      {len(records)} segment-layer records collected")

        # ------------------------------------------------------------------
        # Probe each layer
        # ------------------------------------------------------------------
        print(f"[4/4] Running phoneme probes (speaker-held-out, {args.n_folds}-fold) …")
        layer_results = {}

        for layer_idx in layer_indices:
            print(f"     Layer {layer_idx:2d} | probing …")

            X, phone_ids, _, speakers = records_to_arrays(records, layer_idx)

            # Filter out unknown phones
            valid = phone_ids >= 0
            X, phone_ids, speakers = X[valid], phone_ids[valid], speakers[valid]

            if len(X) < 50:
                print(f"    Layer {layer_idx:2d} | skipped (too few samples: {len(X)})")
                continue

            #subsample if too large (to speed up probing; use fixed seed for reproducibility)
            if len(X) > 30000:
                print(f"    Layer {layer_idx:2d} | subsampling to 30k records for faster probing (full set has {len(X)} records)")
                idx = np.random.RandomState(42).choice(len(X), 30000, replace=False)
                X, phone_ids, speakers = X[idx], phone_ids[idx], speakers[idx]

            result = run_phoneme_probe(X, phone_ids, speakers, layer_idx, args.n_folds)
            layer_results[str(layer_idx)] = result

        all_results[model_name] = layer_results
        del base

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.output_dir) / f"phoneme_probe_{args.split}.json"
    save_results(all_results, str(out_path))
    print(f"\nDone. Results written to {out_path}")

    # Quick summary table to stdout
    print("\n=== Summary: Phoneme Probe Accuracy by Layer ===")
    header = f"{'Layer':>6}" + "".join(f"  {m:>12}" for m in all_results)
    print(header)
    for li in layer_indices:
        row = f"{li:>6}"
        for m in all_results:
            val = all_results[m].get(str(li), {}).get("accuracy", float("nan"))
            row += f"  {val:>12.3f}"
        print(row)


if __name__ == "__main__":
    from typing import Optional
    main()

