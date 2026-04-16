"""
probe_accent.py
Linear probing for accent / L1 identity from Whisper encoder layers.

Usage:
    python probe_accent.py --models baseline --split scripted
    python probe_accent.py --models baseline,no_aux,ctc_aux,feat_aux --split scripted
    python probe_accent.py --models ctc_aux --split scripted --within_phoneme

Output: results/accent_probe/accent_probe_{model_key}_{split}.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import LOCAL_L2ARCTIC_DIR, NUM_L1S, PROBE_PHONES, WHISPER_N_ENCODER_LAYERS
from src.eval.probing.probe_utils import (
    get_or_build_embeddings,
    records_to_arrays,
    save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry
from src.utils.phonology import ARPABET_VOCAB

MAX_PROBE_SAMPLES = 10000

# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_accent_probe(
    X:        np.ndarray,
    l1_ids:   np.ndarray,
    speakers: np.ndarray,
    n_folds:  int = 5,
) -> dict:
    """Speaker-grouped k-fold multinomial logistic regression probe for L1 identity."""
    actual_folds = min(n_folds, len(np.unique(speakers)))
    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n_samples": len(X)}

    all_preds, all_true = [], []
    for tr, te in GroupKFold(n_splits=actual_folds).split(X, l1_ids, speakers):
        scaler = StandardScaler()
        clf = LogisticRegression(
            solver="saga",
            max_iter=1000,
            tol=1e-3,
            C=1.0,
            random_state=42,
        )
        clf.fit(scaler.fit_transform(X[tr]), l1_ids[tr])
        all_preds.extend(clf.predict(scaler.transform(X[te])).tolist())
        all_true.extend(l1_ids[te].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    return {
        "accuracy":  float(accuracy_score(all_true, all_preds)),
        "macro_f1":  float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "n_samples": int(len(all_true)),
    }


def run_within_phoneme_accent_probe(
    X:           np.ndarray,
    phone_ids:   np.ndarray,
    l1_ids:      np.ndarray,
    speakers:    np.ndarray,
    min_samples: int = 30,
    n_folds:     int = 5,
) -> dict:
    """Run accent probe separately for each phone in PROBE_PHONES."""
    per_phone = {}
    for pid in np.unique(phone_ids):
        phone_label = ARPABET_VOCAB[pid]
        if phone_label not in PROBE_PHONES:
            continue
        mask = phone_ids == pid
        if mask.sum() < min_samples:
            continue
        per_phone[phone_label] = run_accent_probe(
            X[mask], l1_ids[mask], speakers[mask], n_folds=n_folds,
        )
    return per_phone


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
    within_phoneme: bool,
    device:         str,
    output_dir:     str,
    split:          str,
) -> None:
    out_path = Path(output_dir) / f"accent_probe_{model_key}_{split}.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    chance = 1.0 / NUM_L1S
    print(f"  Chance baseline = {chance:.3f}")

    records = get_or_build_embeddings(
        model_key, model, processor, utterances, layer_indices, device
    )
    print(f"  {len(records):,} records")

    layer_results = {}
    for layer_idx in layer_indices:
        X, phone_ids, l1_ids, speakers, _ = records_to_arrays(records, layer_idx)

        if len(X) > MAX_PROBE_SAMPLES:
            print(f"  Layer {layer_idx:2d} | downsampling from {len(X):,} to {MAX_PROBE_SAMPLES} samples")
            # stratify by speaker to preserve group structure
            idx = []
            unique_speakers = np.unique(speakers)
            per_spk = MAX_PROBE_SAMPLES // len(unique_speakers)
            rng = np.random.default_rng(42)
            for spk in unique_speakers:
                spk_idx = np.where(speakers == spk)[0]
                chosen = rng.choice(spk_idx, min(per_spk, len(spk_idx)), replace=False)
                idx.extend(chosen)
            idx = np.array(idx)
            X, phone_ids, l1_ids, speakers = X[idx], phone_ids[idx], l1_ids[idx], speakers[idx]

        global_result = run_accent_probe(X, l1_ids, speakers, n_folds=n_folds)
        print(f"  Layer {layer_idx:2d} | acc={global_result['accuracy']:.3f}"
              f"  macro-F1={global_result['macro_f1']:.3f}"
              f"  n={global_result['n_samples']:,}  (chance={chance:.3f})")

        layer_entry = {"global": global_result, "chance_accuracy": chance}

        if within_phoneme:
            wp = run_within_phoneme_accent_probe(
                X, phone_ids, l1_ids, speakers, n_folds=n_folds
            )
            wp_accs = [v["accuracy"] for v in wp.values() if not np.isnan(v["accuracy"])]
            mean_wp = float(np.mean(wp_accs)) if wp_accs else float("nan")
            print(f"             within-phoneme mean acc={mean_wp:.3f}")
            layer_entry["within_phoneme"]          = wp
            layer_entry["mean_within_phoneme_acc"] = mean_wp

        layer_results[str(layer_idx)] = layer_entry

    save_results(layer_results, str(out_path))

    print(f"\n{'Layer':>6}  {'Accuracy':>10}  {'MacroF1':>10}")
    for li in layer_indices:
        acc = layer_results[str(li)]["global"]["accuracy"]
        mf1 = layer_results[str(li)]["global"]["macro_f1"]
        print(f"  {li:>6}  {acc:>10.3f}  {mf1:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser(description="Accent / L1 linear probe on Whisper encoder")
    p.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--models",               default="baseline",
                   help="Comma-separated model keys from get_model_registry")
    p.add_argument("--output_dir",           default="results/accent_probe")
    p.add_argument("--layers",               default=None,
                   help="Comma-separated layer indices (default: all)")
    p.add_argument("--max_utts_per_speaker", type=int, default=100)
    p.add_argument("--n_folds",              type=int, default=5)
    p.add_argument("--split",                default="scripted",
                   choices=["scripted", "spontaneous", "all"])
    p.add_argument("--within_phoneme",       action="store_true")
    args = p.parse_args()

    print(f"=== Accent Probe  device={device}  split={args.split} ===")

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
            within_phoneme = args.within_phoneme,
            device         = device,
            output_dir     = args.output_dir,
            split          = args.split,
        )
        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()