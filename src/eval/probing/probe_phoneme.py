"""
probe_phoneme.py
Linear probing for PHONEME IDENTITY from Whisper encoder layers.

For each encoder layer (0-6) of each model:
  - Trains a logistic regression on segment-level mean-pooled embeddings
  - Predicts ARPAbet phone class (39-way classification)
  - Evaluates accuracy, macro-F1, and per-phone F1
  - Speaker-held-out cross-validation to prevent leakage

Usage:
    # Single model (parallelisable)
    python probe_phoneme.py --models baseline   --split scripted
    python probe_phoneme.py --models ctc_aux    --split scripted

    # Multiple models in one run (sequential)
    python probe_phoneme.py --models baseline,baseline_lora,ctc_aux --split scripted
"""

import argparse
import numpy as np
from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import torch

from src.config import LOCAL_L2ARCTIC_DIR
from src.eval.probing.probe_utils import (
    ARPABET_VOCAB, NUM_PHONES, WHISPER_N_ENCODER_LAYERS,
    build_embedding_dataset, records_to_arrays, save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_phoneme_probe(X, y, groups, layer_idx: int, n_folds: int = 5) -> dict:
    """Speaker-held-out logistic regression probe for phoneme identity."""
    gkf = GroupKFold(n_splits=n_folds)
    all_preds, all_true = [], []

    for _, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[train_idx])
        X_te   = scaler.transform(X[test_idx])
        clf    = SGDClassifier(max_iter=300, loss="log_loss", random_state=42)
        clf.fit(X_tr, y[train_idx])
        all_preds.extend(clf.predict(X_te).tolist())
        all_true.extend(y[test_idx].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc      = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_arr   = f1_score(all_true, all_preds, labels=list(range(NUM_PHONES)),
                        average=None, zero_division=0)
    per_phone_f1 = {ARPABET_VOCAB[i]: float(f1_arr[i]) for i in range(NUM_PHONES)}

    print(f"  Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={macro_f1:.3f}")
    return {
        "accuracy":     float(acc),
        "macro_f1":     float(macro_f1),
        "per_phone_f1": per_phone_f1,
        "n_samples":    int(len(all_true)),
        "n_folds":      n_folds,
    }


# ---------------------------------------------------------------------------
# Per-model probe runner
# ---------------------------------------------------------------------------

def probe_model(model_key, model, processor, utterances, layer_indices,
                    n_folds, device, output_dir, split):
    """Run the full phoneme probe for a single model and save its own JSON."""
    out_path = Path(output_dir) / f"phoneme_probe_{model_key}_{split}.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    print(f"  Extracting hidden states …")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records)} segment-layer records collected")

    layer_results = {}
    for layer_idx in layer_indices:
        X, phone_ids, _, speakers = records_to_arrays(records, layer_idx)

        # Filter unknown phones
        valid = phone_ids >= 0
        X, phone_ids, speakers = X[valid], phone_ids[valid], speakers[valid]

        if len(X) < 50:
            print(f"  Layer {layer_idx:2d} | skipped (too few samples: {len(X)})")
            continue

        if len(X) > 30000:
            print(f"  Layer {layer_idx:2d} | subsampling 30k from {len(X)} records")
            idx = np.random.RandomState(42).choice(len(X), 30000, replace=False)
            X, phone_ids, speakers = X[idx], phone_ids[idx], speakers[idx]

        result = run_phoneme_probe(X, phone_ids, speakers, layer_idx, n_folds)
        layer_results[str(layer_idx)] = result

    save_results(layer_results, str(out_path))
    print(f"  Saved → {out_path}")

    # Summary
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
    parser = argparse.ArgumentParser(description="Phoneme linear probe on Whisper encoder")
    parser.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--models",               default="baseline",
                        help=f"Comma-separated model keys, options: {get_model_registry(device).keys()}")
    parser.add_argument("--split",                default="scripted",
                        choices=["scripted", "spontaneous", "all"])
    parser.add_argument("--output_dir",           default="results/phoneme_probe")
    parser.add_argument("--layers",               default=None,
                        help="Comma-separated layer indices, default=all")
    parser.add_argument("--max_utts_per_speaker", type=int, default=50)
    parser.add_argument("--n_folds",              type=int, default=5)
    args = parser.parse_args()

    print(f"=== Phoneme Probe === device={device}")

    layer_indices = (
        [int(x) for x in args.layers.split(",")]
        if args.layers else list(range(WHISPER_N_ENCODER_LAYERS + 1))
    )
    model_keys = [k.strip() for k in args.models.split(",")]
    registry   = get_model_registry(device)

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key: '{key}'. Available: {list(registry.keys())}")

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
            model_key     = key,
            model         = model,
            processor     = processor,
            utterances    = utterances,
            layer_indices = layer_indices,
            n_folds       = args.n_folds,
            device        = device,
            output_dir    = args.output_dir,
            split         = args.split,
        )
        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
