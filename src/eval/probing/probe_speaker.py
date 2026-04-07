"""
probe_speaker.py
Linear probing for SPEAKER IDENTITY from Whisper encoder layers.

Purpose (control probe):
    If the accent probe shows high L1 accuracy, it could simply be because
    speaker identity is recoverable at that layer and each speaker has a
    unique accent. The speaker probe lets us distinguish between:

        (a) The model encodes *speaker-level* acoustic idiosyncrasies
            → speaker probe ≈ accent probe accuracy
        (b) The model encodes *group-level* L1 accent structure
            → speaker probe >> accent probe accuracy (speaker more decodable)
               OR speaker probe < accent probe (accent is encoded more abstractly)

    Case (b) is the interesting one for your thesis. A layer where accent
    probe accuracy is HIGHER than would be expected from speaker-level cues
    is encoding genuine group-level phonological accent information.

Usage:
    # Single model (parallelisable)
    python probe_speaker.py --models baseline  --split scripted
    python probe_speaker.py --models ctc_aux   --split scripted

    # Multiple models in one run
    python probe_speaker.py --models baseline,baseline_lora,ctc_aux --split scripted

    # With accent-beyond-speaker analysis (point at per-model accent probe JSON)
    python probe_speaker.py --models ctc_aux --split scripted \
        --accent_results results/accent_probe/accent_probe_ctc_aux_scripted.json
"""

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import torch

from src.eval.probing.probe_utils import (
    WHISPER_N_ENCODER_LAYERS, NUM_L1S,
    build_embedding_dataset, records_to_arrays, save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry
from src.config import LOCAL_L2ARCTIC_DIR


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_speaker_probe(X, speakers, layer_idx, n_folds=5) -> dict:
    """Leave-one-speaker-out logistic regression for speaker identity."""
    le = LabelEncoder()
    speaker_ids = le.fit_transform(speakers)
    n_speakers  = len(le.classes_)
    actual_folds = min(n_folds, n_speakers)

    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "chance_accuracy": 1.0 / n_speakers,
                "n_speakers": n_speakers, "n_samples": len(X)}

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds, all_true = [], []

    for _, (train_idx, test_idx) in enumerate(gkf.split(X, speaker_ids, speakers)):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[train_idx])
        X_te   = scaler.transform(X[test_idx])
        clf    = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, speaker_ids[train_idx])
        all_preds.extend(clf.predict(X_te).tolist())
        all_true.extend(speaker_ids[test_idx].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    chance    = 1.0 / n_speakers

    acc = accuracy_score(all_true, all_preds)
    mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"  Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={mf1:.3f}"
          f"  (chance={chance:.3f}, n_spk={n_speakers})")

    return {
        "accuracy":        float(acc),
        "macro_f1":        float(mf1),
        "chance_accuracy": float(chance),
        "n_speakers":      int(n_speakers),
        "n_samples":       int(len(all_true)),
    }


def compute_accent_beyond_speaker(speaker_acc, accent_acc, n_speakers, n_l1s) -> float:
    """
    Heuristic residual: how much accent decodability remains after
    factoring out what speaker-level cues alone would predict.
    Positive = layer encodes genuine group-level accent structure.
    """
    chance_l1  = 1.0 / n_l1s
    chance_spk = 1.0 / n_speakers
    accent_residual  = accent_acc  - chance_l1
    speaker_residual = speaker_acc - chance_spk
    if speaker_residual <= 0:
        return float(accent_residual)
    return float(accent_residual - speaker_residual * (chance_l1 / chance_spk))


# ---------------------------------------------------------------------------
# Per-model probe runner
# ---------------------------------------------------------------------------

def probe_model(model_key, model, processor, utterances, layer_indices,
                    n_folds, device, output_dir, split, accent_results=None):
    """Run the full speaker probe for a single model and save its own JSON."""
    out_path = Path(output_dir) / f"speaker_probe_{model_key}_{split}.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    print(f"  Extracting hidden states …")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records)} records")

    layer_results = {}
    for layer_idx in layer_indices:
        X, _, _, speakers = records_to_arrays(records, layer_idx)

        if len(X) > 30000:
            idx = np.random.RandomState(42).choice(len(X), 30000, replace=False)
            X, speakers = X[idx], speakers[idx]

        result = run_speaker_probe(X, speakers, layer_idx, n_folds)

        # Accent-beyond-speaker — reads from the per-model accent probe JSON
        # accent_results here is already the flat { layer: { global: {...} } } dict
        if accent_results is not None:
            accent_entry = accent_results.get(str(layer_idx), {})
            # handle both flat and model-key-wrapped formats
            if model_key in accent_entry:
                accent_entry = accent_entry[model_key].get(str(layer_idx), {})
            accent_acc = accent_entry.get("global", {}).get("accuracy", float("nan"))
            if not np.isnan(accent_acc):
                delta = compute_accent_beyond_speaker(
                    result["accuracy"], accent_acc,
                    result["n_speakers"], NUM_L1S,
                )
                result["accent_beyond_speaker"] = delta
                print(f"             accent-beyond-speaker Δ = {delta:+.3f}")

        layer_results[str(layer_idx)] = result

    save_results(layer_results, str(out_path))
    print(f"  Saved → {out_path}")

    print(f"  {'Layer':>6}  {'Accuracy':>10}  {'MacroF1':>10}")
    for li in layer_indices:
        acc = layer_results[str(li)]["accuracy"]
        mf1 = layer_results[str(li)]["macro_f1"]
        print(f"  {li:>6}  {acc:>10.3f}  {mf1:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Speaker identity linear probe on Whisper encoder")
    parser.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--models",               default="baseline",
                        help=f"Comma-separated model keys, options: {get_model_registry(device).keys()}")
    parser.add_argument("--split",                default="scripted",
                        choices=["scripted", "spontaneous", "all"])
    parser.add_argument("--output_dir",           default="results/speaker_probe")
    parser.add_argument("--layers",               default=None)
    parser.add_argument("--max_utts_per_speaker", type=int, default=50)
    parser.add_argument("--n_folds",              type=int, default=5)
    parser.add_argument("--accent_results",       default=None,
                        help="Path to accent_probe_{model_key}_{split}.json for "
                             "accent-beyond-speaker computation. If provided, must "
                             "match the single model key being evaluated.")
    args = parser.parse_args()

    print(f"=== Speaker Probe === device={device}")

    layer_indices = (
        [int(x) for x in args.layers.split(",")]
        if args.layers else list(range(WHISPER_N_ENCODER_LAYERS + 1))
    )
    model_keys = [k.strip() for k in args.models.split(",")]
    registry   = get_model_registry(device)

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key: '{key}'. Available: {list(registry.keys())}")

    # Load accent results once if provided (only meaningful for single-model runs)
    accent_results = None
    if args.accent_results:
        with open(args.accent_results) as f:
            raw = json.load(f)
        # New per-model format: flat { layer: {...} }
        # Old merged format: { model_key: { layer: {...} } }
        # Detect by checking if first value is a dict with numeric string keys
        first_val = next(iter(raw.values()))
        if isinstance(first_val, dict) and "global" in first_val:
            accent_results = raw          # flat format
        else:
            # merged format — unwrap the first (and presumably only) model
            accent_results = next(iter(raw.values()))
        print(f"  Loaded accent results from {args.accent_results}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading utterances (split={args.split}) …")
    utterances = load_probe_utterances(
        local_root=args.data_root, split=args.split,
        max_utts_per_speaker=args.max_utts_per_speaker, speakers={"all"},
    )
    print(f"    Found {len(utterances)} utterances")

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
            split          = args.split,
            accent_results = accent_results,
        )
        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
