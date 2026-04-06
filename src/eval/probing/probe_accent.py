"""
probe_accent.py
Linear probing for ACCENT / L1 IDENTITY from Whisper encoder layers.

For each encoder layer (0-6) of each model:
  - Trains a logistic regression on segment-level embeddings (same embeddings
    as the phoneme probe, but now predicting L1 rather than phoneme)
  - Uses speaker-held-out cross-validation (same speaker never in train AND test)
  - Also measures how much accent information is recoverable when the probe
    is conditioned on phoneme identity (i.e. within-phoneme accent separability)

The key diagnostic question:
    "At which layers is L1 accent linearly decodable,
     and does this overlap with where phoneme info is strongest?"

Usage:
    python probe_accent.py \
        --data_root /path/to/l2arctic \
        --baseline_model openai/whisper-small \
        --lora_model   /path/to/lora-checkpoint \
        --split        scripted \
        --output_dir   results/accent_probe \
        [--max_utts 500]
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.eval.probing.probe_utils import (
    L1_GROUPS, L1_2_ID, NUM_L1S, ARPABET_VOCAB, WHISPER_N_ENCODER_LAYERS,
    build_embedding_dataset, records_to_arrays, save_results,
    SPEAKER_L1,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.config import LOCAL_L2ARCTIC_DIR



# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_accent_probe(X, l1_ids, speakers, layer_idx: int, n_folds: int = 5,
                     label: str = "global") -> dict:
    """
    Speaker-held-out L1 classification probe.
    `label` is just for display (e.g. 'global' or phone-specific).
    """
    gkf = GroupKFold(n_splits=n_folds)
    all_preds, all_true = [], []

    # Need at least n_folds unique speakers
    unique_spk = np.unique(speakers)
    actual_folds = min(n_folds, len(unique_spk))
    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n_samples": len(X)}

    gkf_actual = GroupKFold(n_splits=actual_folds)
    for _, (train_idx, test_idx) in enumerate(gkf_actual.split(X, l1_ids, speakers)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = l1_ids[train_idx], l1_ids[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", n_jobs=-1,
        )
        clf.fit(X_tr_s, y_tr)
        all_preds.extend(clf.predict(X_te_s).tolist())
        all_true.extend(y_te.tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    return {
        "accuracy":  float(acc),
        "macro_f1":  float(mf1),
        "n_samples": int(len(all_true)),
    }


def run_within_phoneme_accent_probe(
    X, phone_ids, l1_ids, speakers, layer_idx: int,
    min_samples_per_phone: int = 30, n_folds: int = 5,
) -> dict:
    """
    For each phoneme class, run an accent probe on just that phoneme's embeddings.
    This tests whether L1 can be decoded even when phoneme class is controlled for —
    i.e. whether accent and phoneme representations are entangled at this layer.

    Returns: dict mapping phone label → {accuracy, macro_f1, n_samples}
    """
    per_phone = {}
    unique_phones = np.unique(phone_ids)
    for pid in unique_phones:
        mask = phone_ids == pid
        if mask.sum() < min_samples_per_phone:
            continue
        phone_label = ARPABET_VOCAB[pid]
        result = run_accent_probe(
            X[mask], l1_ids[mask], speakers[mask],
            layer_idx, n_folds=n_folds,
            label=phone_label,
        )
        per_phone[phone_label] = result

    return per_phone


def chance_accuracy(n_classes: int) -> float:
    return 1.0 / n_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Accent/L1 linear probe on Whisper encoder")
    parser.add_argument("--data_root",       default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--baseline_model",  default="openai/whisper-small")
    parser.add_argument("--lora_model",      default="models/baseline_loraft")
    parser.add_argument("--split",           default="scripted",
                        choices=["scripted","spontaneous","all"])
    parser.add_argument("--output_dir",      default="results/accent_probe")
    parser.add_argument("--layers",          default=None)
    parser.add_argument("--max_utts",        type=int, default=None)
    parser.add_argument("--n_folds",         type=int, default=5)
    parser.add_argument("--within_phoneme",  action="store_true",
                        help="Also run per-phoneme accent probes (more expensive)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Accent Probe ===")
    print(f"Running on device: {device}")

    layer_indices = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(WHISPER_N_ENCODER_LAYERS + 1))
    )

    print(f"\n[1/4] Loading utterances (split={args.split}) …")
    utterances = load_probe_utterances(
        local_root=args.data_root,
        split=args.split,
        max_utts=args.max_utts,
    )
    print(f"      Found {len(utterances)} utterances")

    models_to_eval = {"baseline": args.baseline_model}
    if args.lora_model:
        models_to_eval["lora"] = args.lora_model

    all_results = {}
    chance = chance_accuracy(NUM_L1S)
    print(f"      Chance accuracy (1/{NUM_L1S} L1s) = {chance:.3f}")

    for model_name, model_path in models_to_eval.items():
        print(f"\n[2/4] Model: {model_name}")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        if model_name == "lora" and args.lora_model:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, model_path)
            base = base.merge_and_unload()
        base = base.to(device)

        print("[3/4] Extracting hidden states …")
        records = build_embedding_dataset(
            model=base, processor=processor,
            utterances=utterances, layer_indices=layer_indices,
            device=device,
        )
        print(f"      {len(records)} records")

        print(f"[4/4] Running accent probes …")
        layer_results = {}

        for layer_idx in layer_indices:
            X, phone_ids, l1_ids, speakers = records_to_arrays(records, layer_idx)

            # --- Global accent probe ---
            global_result = run_accent_probe(
                X, l1_ids, speakers, layer_idx, args.n_folds
            )
            print(f"    Layer {layer_idx:2d} | acc={global_result['accuracy']:.3f}"
                  f"  macro-F1={global_result['macro_f1']:.3f}"
                  f"  (chance={chance:.3f})")

            layer_entry = {
                "global":         global_result,
                "chance_accuracy": chance,
            }

            # --- Within-phoneme accent probe (optional, expensive) ---
            if args.within_phoneme:
                wp = run_within_phoneme_accent_probe(
                    X, phone_ids, l1_ids, speakers, layer_idx, n_folds=args.n_folds
                )
                # Compute mean within-phoneme accuracy (where computable)
                wp_accs = [v["accuracy"] for v in wp.values()
                           if not np.isnan(v["accuracy"])]
                mean_wp = float(np.mean(wp_accs)) if wp_accs else float("nan")
                print(f"             within-phoneme mean acc={mean_wp:.3f}")
                layer_entry["within_phoneme"] = wp
                layer_entry["mean_within_phoneme_acc"] = mean_wp

            layer_results[str(layer_idx)] = layer_entry

        all_results[model_name] = layer_results
        del base

    out_path = Path(args.output_dir) / f"accent_probe_{args.split}.json"
    save_results(all_results, str(out_path))
    print(f"\nDone. Results → {out_path}")

    # Summary table
    print("\n=== Summary: Accent Probe Accuracy by Layer ===")
    header = f"{'Layer':>6}" + "".join(f"  {m:>14}" for m in all_results)
    print(header)
    for li in layer_indices:
        row = f"{li:>6}"
        for m in all_results:
            val = all_results[m].get(str(li), {}).get("global", {}).get("accuracy", float("nan"))
            row += f"  {val:>14.3f}"
        print(row)
    print(f"  (chance = {chance:.3f})")


if __name__ == "__main__":
    main()

