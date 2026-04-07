"""
probe_accent.py
Linear probing for ACCENT / L1 IDENTITY from Whisper encoder layers.

Key changes vs previous version:
  - --models  accepts a comma-separated list of keys from MODEL_REGISTRY
    (e.g. --models baseline,baseline_lora,ctc_aux)
  - Each model is saved to its own JSON file so runs are fully independent
    and can be parallelised across jobs
  - Output: results/accent_probe/accent_probe_{model_key}_{split}.json

Usage:
    # Single model (parallelisable)
    python probe_accent.py --models baseline   --split scripted
    python probe_accent.py --models ctc_aux    --split scripted

    # Multiple models in one run (sequential)
    python probe_accent.py --models baseline,baseline_lora,ctc_aux --split scripted

    # With within-phoneme accent probe
    python probe_accent.py --models ctc_aux --split scripted --within_phoneme
"""

import argparse
import numpy as np
from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import torch

from src.eval.probing.probe_utils import (
    ARPABET_VOCAB, WHISPER_N_ENCODER_LAYERS,
    build_embedding_dataset, records_to_arrays, save_results,
)
from src.utils.model_loader import get_model_registry
from src.utils.load_l2arctic import load_probe_utterances
from src.config import LOCAL_L2ARCTIC_DIR, NUM_L1S, PROBE_PHONES


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_accent_probe(X, l1_ids, speakers, layer_idx: int,
                     n_folds: int = 5, label: str = "global") -> dict:
    all_preds, all_true = [], []
    unique_spk   = np.unique(speakers)
    actual_folds = min(n_folds, len(unique_spk))
    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n_samples": len(X)}

    gkf = GroupKFold(n_splits=actual_folds)
    for _, (train_idx, test_idx) in enumerate(gkf.split(X, l1_ids, speakers)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf  = SGDClassifier(max_iter=300, loss="log_loss", random_state=42)
        clf.fit(X_tr, l1_ids[train_idx])
        all_preds.extend(clf.predict(X_te).tolist())
        all_true.extend(l1_ids[test_idx].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    return {
        "accuracy":  float(accuracy_score(all_true, all_preds)),
        "macro_f1":  float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "n_samples": int(len(all_true)),
    }


def run_within_phoneme_accent_probe(X, phone_ids, l1_ids, speakers, layer_idx: int,
                                    min_samples: int = 30, n_folds: int = 5) -> dict:
    per_phone = {}
    for pid in np.unique(phone_ids):
        phone_label = ARPABET_VOCAB[pid]
        if phone_label not in PROBE_PHONES:
            continue
        mask = phone_ids == pid
        if mask.sum() < min_samples:
            continue
        per_phone[phone_label] = run_accent_probe(
            X[mask], l1_ids[mask], speakers[mask], layer_idx, n_folds=n_folds,
        )
    return per_phone


# ---------------------------------------------------------------------------
# Per-model probe runner
# ---------------------------------------------------------------------------

def probe_model(model_key, model, processor, utterances, layer_indices,
                    n_folds, within_phoneme, device, output_dir, split):
    """Run the full accent probe for a single model and save its own JSON."""
    out_path = Path(output_dir) / f"accent_probe_{model_key}_{split}.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists — delete to re-run")
        return

    chance = 1.0 / NUM_L1S
    print(f"  Chance accuracy (1/{NUM_L1S}) = {chance:.3f}")

    print(f"  Extracting hidden states …")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records)} records")

    layer_results = {}
    for layer_idx in layer_indices:
        X, phone_ids, l1_ids, speakers = records_to_arrays(records, layer_idx)

        # Subsample if too large — stratified by speaker
        if len(X) > 30000:
            from sklearn.model_selection import train_test_split
            _, X, _, phone_ids, _, l1_ids, _, speakers = train_test_split(
                X, phone_ids, l1_ids, speakers,
                test_size=30000, random_state=42, stratify=speakers,
            )

        global_result = run_accent_probe(X, l1_ids, speakers, layer_idx, n_folds)
        print(f"  Layer {layer_idx:2d} | acc={global_result['accuracy']:.3f}"
              f"  macro-F1={global_result['macro_f1']:.3f}"
              f"  (chance={chance:.3f})")

        layer_entry = {"global": global_result, "chance_accuracy": chance}

        if within_phoneme:
            wp      = run_within_phoneme_accent_probe(X, phone_ids, l1_ids, speakers, layer_idx, n_folds=n_folds)
            wp_accs = [v["accuracy"] for v in wp.values() if not np.isnan(v["accuracy"])]
            mean_wp = float(np.mean(wp_accs)) if wp_accs else float("nan")
            print(f"             within-phoneme mean acc={mean_wp:.3f}")
            layer_entry["within_phoneme"]          = wp
            layer_entry["mean_within_phoneme_acc"] = mean_wp

        layer_results[str(layer_idx)] = layer_entry

    save_results(layer_results, str(out_path))
    print(f"  Saved → {out_path}")

    # Summary
    print(f"  {'Layer':>6}  {'Accuracy':>10}  {'MacroF1':>10}")
    for li in layer_indices:
        acc = layer_results[str(li)]["global"]["accuracy"]
        mf1 = layer_results[str(li)]["global"]["macro_f1"]
        print(f"  {li:>6}  {acc:>10.3f}  {mf1:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Accent/L1 linear probe on Whisper encoder")
    parser.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--models",               default="baseline",
                        help=f"Comma-separated model keys, options: {get_model_registry(device).keys()}")
    parser.add_argument("--split",                default="scripted",
                        choices=["scripted", "spontaneous", "all"])
    parser.add_argument("--output_dir",           default="results/accent_probe")
    parser.add_argument("--layers",               default=None,
                        help="Comma-separated layer indices, default=all")
    parser.add_argument("--max_utts_per_speaker", type=int, default=50)
    parser.add_argument("--n_folds",              type=int, default=5)
    parser.add_argument("--within_phoneme",       action="store_true")
    args = parser.parse_args()

    print(f"=== Accent Probe === device={device}")

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
