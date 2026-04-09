"""
probe_speaker.py
Linear probing for speaker identity from Whisper encoder layers.

Purpose — control probe:
    A layer where accent probe accuracy is high but *not* explained by
    speaker-level cues is encoding genuine group-level L1 accent structure.
    The `accent_beyond_speaker` metric quantifies this residual.

Usage:
    python probe_speaker.py --models baseline --split scripted
    python probe_speaker.py --models baseline,baseline_lora,ctc_aux --split scripted
    python probe_speaker.py --models ctc_aux --split scripted \
        --accent_results results/accent_probe/accent_probe_ctc_aux_scripted.json

Output: results/speaker_probe/speaker_probe_{model_key}_{split}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import LOCAL_L2ARCTIC_DIR, NUM_L1S, WHISPER_N_ENCODER_LAYERS
from src.eval.probing.probe_utils import (
    build_embedding_dataset,
    records_to_arrays,
    save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------


def run_speaker_probe(
    X:        np.ndarray,
    speakers: np.ndarray,
    layer_idx: int,
    n_folds:  int = 5,
) -> dict:
    """Speaker-grouped k-fold probe for speaker identity."""
    le          = LabelEncoder()
    speaker_ids = le.fit_transform(speakers)
    n_speakers  = len(le.classes_)
    actual_folds = min(n_folds, n_speakers)

    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "chance_accuracy": 1.0 / n_speakers,
                "n_speakers": n_speakers, "n_samples": len(X)}

    all_preds, all_true = [], []
    for tr, te in GroupKFold(n_splits=actual_folds).split(X, speaker_ids, speakers):
        scaler = StandardScaler()
        clf    = SGDClassifier(max_iter=300, loss="log_loss", random_state=42)
        clf.fit(scaler.fit_transform(X[tr]), speaker_ids[tr])
        all_preds.extend(clf.predict(scaler.transform(X[te])).tolist())
        all_true.extend(speaker_ids[te].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    chance    = 1.0 / n_speakers

    acc = float(accuracy_score(all_true, all_preds))
    mf1 = float(f1_score(all_true, all_preds, average="macro", zero_division=0))
    print(f"  Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={mf1:.3f}"
          f"  (chance={chance:.3f}  n_spk={n_speakers})")

    return {
        "accuracy":        acc,
        "macro_f1":        mf1,
        "chance_accuracy": float(chance),
        "n_speakers":      int(n_speakers),
        "n_samples":       int(len(all_true)),
    }


def accent_beyond_speaker(
    speaker_acc: float,
    accent_acc:  float,
    n_speakers:  int,
    n_l1s:       int,
) -> float:
    """
    Residual accent decodability after factoring out speaker-level cues.
    Positive values indicate genuine group-level accent encoding.
    """
    chance_l1  = 1.0 / n_l1s
    chance_spk = 1.0 / n_speakers
    accent_residual  = accent_acc  - chance_l1
    speaker_residual = speaker_acc - chance_spk
    if speaker_residual <= 0:
        return float(accent_residual)
    return float(accent_residual - speaker_residual * (chance_l1 / chance_spk))


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------


def probe_model(
    model_key:      str,
    model,
    processor,
    utterances:     list,
    layer_indices:  list[int],
    n_folds:        int,
    device:         str,
    output_dir:     str,
    split:          str,
    accent_results: dict | None = None,
) -> None:
    out_path = Path(output_dir) / f"speaker_probe_{model_key}_{split}.json"
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
        X, _, _, speakers, _ = records_to_arrays(records, layer_idx)

        result = run_speaker_probe(X, speakers, layer_idx, n_folds)

        if accent_results is not None:
            accent_acc = accent_results.get(str(layer_idx), {}).get(
                "global", {}
            ).get("accuracy", float("nan"))
            if not np.isnan(accent_acc):
                delta = accent_beyond_speaker(
                    result["accuracy"], accent_acc,
                    result["n_speakers"], NUM_L1S,
                )
                result["accent_beyond_speaker"] = delta
                print(f"             accent-beyond-speaker Δ={delta:+.3f}")

        layer_results[str(layer_idx)] = result

    save_results(layer_results, str(out_path))
    print(f"  Saved → {out_path}")

    print(f"\n  {'Layer':>6}  {'Accuracy':>10}  {'MacroF1':>10}")
    for li in layer_indices:
        r = layer_results[str(li)]
        print(f"  {li:>6}  {r['accuracy']:>10.3f}  {r['macro_f1']:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser(description="Speaker identity linear probe on Whisper encoder")
    p.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--models",               default="baseline",
                   help="Comma-separated model keys from MODEL_REGISTRY")
    p.add_argument("--output_dir",           default="results/speaker_probe")
    p.add_argument("--layers",               default=None,
                   help="Comma-separated layer indices (default: all)")
    p.add_argument("--max_utts_per_speaker", type=int, default=100)
    p.add_argument("--n_folds",              type=int, default=5)
    p.add_argument("--accent_results",       default=None,
                   help="Path to accent_probe_{model_key}_{split}.json for "
                        "accent-beyond-speaker computation")
    args = p.parse_args()

    print(f"=== Speaker Probe  device={device} ===")

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

    # Load accent results if provided — expect flat {layer: {global: {...}}} format
    accent_results = None
    if args.accent_results:
        with open(args.accent_results) as f:
            accent_results = json.load(f)
        print(f"  Loaded accent results from {args.accent_results}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading utterances ...")
    utterances = load_probe_utterances(
        local_root           = args.data_root,
        max_utts_per_speaker = args.max_utts_per_speaker,
    )
    print(f"  {len(utterances):,} utterances loaded")

    for key in model_keys:
        print(f"\n[Model: {key}]")
        model, processor = registry[key]["loader"]()
        probe_model(
            model_key      = key,
            model          = model,
            processor      = processor,
            utterances     = utterances,
            layer_indices  = layer_indices,
            n_folds        = args.n_folds,
            device         = device,
            output_dir     = args.output_dir,
            split          = "scripted",
            accent_results = accent_results,
        )
        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
