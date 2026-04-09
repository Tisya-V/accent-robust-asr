"""
probe_phoneme.py
Linear probing for phoneme identity from Whisper encoder layers.

Usage:
    python probe_phoneme.py --models baseline --split scripted
    python probe_phoneme.py --models baseline,baseline_lora,ctc_aux --split scripted

Output: results/phoneme_probe/phoneme_probe_{model_key}_{split}.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import LOCAL_L2ARCTIC_DIR, WHISPER_N_ENCODER_LAYERS
from src.eval.probing.probe_utils import (
    build_embedding_dataset,
    records_to_arrays,
    save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry
from src.utils.phonology import ARPABET_VOCAB, NUM_PHONES


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------


def run_phoneme_probe(
    X:        np.ndarray,
    y:        np.ndarray,
    groups:   np.ndarray,
    layer_idx: int,
    n_folds:  int = 5,
) -> dict:
    """Speaker-grouped k-fold logistic regression probe for phoneme identity."""
    actual_folds = min(n_folds, len(np.unique(groups)))
    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "per_phone_f1": {}, "n_samples": len(X)}

    all_preds, all_true = [], []
    for tr, te in GroupKFold(n_splits=actual_folds).split(X, y, groups):
        scaler = StandardScaler()
        clf    = SGDClassifier(max_iter=300, loss="log_loss", random_state=42)
        clf.fit(scaler.fit_transform(X[tr]), y[tr])
        all_preds.extend(clf.predict(scaler.transform(X[te])).tolist())
        all_true.extend(y[te].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc      = float(accuracy_score(all_true, all_preds))
    macro_f1 = float(f1_score(all_true, all_preds, average="macro", zero_division=0))
    f1_per   = f1_score(all_true, all_preds, labels=list(range(NUM_PHONES)),
                        average=None, zero_division=0)
    per_phone_f1 = {ARPABET_VOCAB[i]: float(f1_per[i]) for i in range(NUM_PHONES)}

    print(f"  Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={macro_f1:.3f}"
          f"  n={len(all_true):,}")
    return {
        "accuracy":     acc,
        "macro_f1":     macro_f1,
        "per_phone_f1": per_phone_f1,
        "n_samples":    int(len(all_true)),
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------


def probe_model(
    model_key:     str,
    model,
    processor,
    utterances:    list,
    layer_indices: list[int],
    n_folds:       int,
    device:        str,
    output_dir:    str,
    split:         str,
) -> None:
    out_path = Path(output_dir) / f"phoneme_probe_{model_key}_{split}.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    print("  Extracting hidden states ...")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records):,} records extracted")

    layer_results = {}
    for layer_idx in layer_indices:
        print(f"\nProbing layer {layer_idx}...")
        X, phone_ids, _, speakers, _ = records_to_arrays(records, layer_idx)

        valid = phone_ids >= 0
        X, phone_ids, speakers = X[valid], phone_ids[valid], speakers[valid]

        if len(X) < 50:
            print(f"  Layer {layer_idx:2d} | skipped (too few samples: {len(X)})")
            continue

        layer_results[str(layer_idx)] = run_phoneme_probe(
            X, phone_ids, speakers, layer_idx, n_folds,
        )

    save_results(layer_results, str(out_path))
    print(f"  Saved → {out_path}")

    print(f"{'Layer':>6}  {'Accuracy':>10}  {'MacroF1':>10}")
    for li in layer_indices:
        r = layer_results.get(str(li))
        if r:
            print(f"  {li:>6}  {r['accuracy']:>10.3f}  {r['macro_f1']:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser(description="Phoneme linear probe on Whisper encoder")
    p.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--models",               default="baseline",
                   help="Comma-separated model keys from MODEL_REGISTRY")
    p.add_argument("--output_dir",           default="results/phoneme_probe")
    p.add_argument("--layers",               default=None,
                   help="Comma-separated layer indices (default: all)")
    p.add_argument("--max_utts_per_speaker", type=int, default=100)
    p.add_argument("--n_folds",              type=int, default=5)
    args = p.parse_args()

    print(f"=== Phoneme Probe  device={device} ===")

    registry      = get_model_registry(device)
    model_keys    = [k.strip() for k in args.models.split(",")]
    layer_indices = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(WHISPER_N_ENCODER_LAYERS + 1))
    )

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key '{key}'. Available: {sorted(registry)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading utterances...")
    utterances = load_probe_utterances(
        local_root           = args.data_root,
        max_utts_per_speaker = args.max_utts_per_speaker,
    )
    print(f"  {len(utterances):,} utterances loaded")

    for key in model_keys:
        print(f"\n[Model: {key}]")
        model, processor = registry[key]["loader"]()
        probe_model(
            model_key     = key,
            model         = model,
            processor     = processor,
            utterances    = utterances,
            layer_indices = layer_indices,
            n_folds       = args.n_folds,
            device        = device,
            output_dir    = args.output_dir,
            split         = "scripted",
        )
        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
