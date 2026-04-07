"""
probe_speaker.py
Linear probing for SPEAKER IDENTITY from Whisper encoder layers.

Purpose (control probe):
    If the accent probe shows high L1 accuracy, it could simply be because
    speaker identity is recoverable at that layer and each speaker has a
    unique accent.  The speaker probe lets us distinguish between:

        (a) The model encodes *speaker-level* acoustic idiosyncrasies
            → speaker probe ≈ accent probe accuracy
        (b) The model encodes *group-level* L1 accent structure
            → speaker probe >> accent probe accuracy (speaker more decodable)
               OR speaker probe < accent probe (accent is encoded more abstractly)

    Case (b) is the interesting one for your thesis.  A layer where accent
    probe accuracy is HIGHER than would be expected from speaker-level cues
    is encoding genuine group-level phonological accent information.

We also compute a simple "accent-beyond-speaker" metric:
    Δ = accent_acc - (accent_acc_expected_from_speaker_acc)

Usage:
    python probe_speaker.py \
        --data_root /path/to/l2arctic \
        --baseline_model openai/whisper-small \
        --lora_model   /path/to/lora-checkpoint \
        --split        scripted \
        --output_dir   results/speaker_probe \
        [--max_utts 500]
"""

import argparse
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.eval.probing.probe_utils import (
    WHISPER_N_ENCODER_LAYERS, SPEAKER_L1, NUM_L1S,
    build_embedding_dataset, records_to_arrays, save_results,
)

from src.utils.load_l2arctic import load_probe_utterances
from src.config import LOCAL_L2ARCTIC_DIR


# ---------------------------------------------------------------------------
# Speaker probe
# ---------------------------------------------------------------------------

def run_speaker_probe(
    X: np.ndarray,
    speakers: np.ndarray,       # string speaker IDs
    layer_idx: int,
    n_folds: int = 5,
) -> dict:
    """
    Leave-one-speaker-out logistic regression for speaker identity.
    Uses GroupKFold so each speaker appears in test exactly once.
    """
    le = LabelEncoder()
    speaker_ids = le.fit_transform(speakers)
    n_speakers = len(le.classes_)

    # Use all speakers as folds (leave-one-speaker-out) up to n_folds
    actual_folds = min(n_folds, n_speakers)
    if actual_folds < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "n_speakers": n_speakers, "n_samples": len(X)}

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds, all_true = [], []

    for _, (train_idx, test_idx) in enumerate(gkf.split(X, speaker_ids, speakers)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = speaker_ids[train_idx], speaker_ids[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=1000, C=1.0,
        )
        clf.fit(X_tr_s, y_tr)
        all_preds.extend(clf.predict(X_te_s).tolist())
        all_true.extend(y_te.tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    chance = 1.0 / n_speakers

    print(f"    Layer {layer_idx:2d} | acc={acc:.3f}  macro-F1={mf1:.3f}"
          f"  (chance={chance:.3f}, n_spk={n_speakers})")

    return {
        "accuracy":       float(acc),
        "macro_f1":       float(mf1),
        "chance_accuracy": float(chance),
        "n_speakers":     int(n_speakers),
        "n_samples":      int(len(all_true)),
    }


def compute_accent_beyond_speaker(
    speaker_acc: float,
    accent_acc: float,
    n_speakers: int,
    n_l1s: int,
) -> float:
    """
    A simple heuristic:
    If speaker identity were the *only* thing driving accent decodability,
    then accent accuracy should scale as n_l1s/n_speakers (since each L1
    group has ~2 speakers, so a speaker-level signal gives ~2/24 ≈ 8% of
    speakers correctly which maps to roughly 1/n_l1s accent accuracy by chance).

    We compute:
        accent_residual = accent_acc - chance_l1
    and
        speaker_residual = speaker_acc - chance_spk

    If accent_residual > 0 after factoring out speaker_residual,
    the layer encodes genuine group-level accent information.
    """
    chance_l1  = 1.0 / n_l1s
    chance_spk = 1.0 / n_speakers
    accent_residual = accent_acc  - chance_l1
    speaker_residual = speaker_acc - chance_spk
    # Proportion of accent signal not explained by speaker-level cues
    if speaker_residual <= 0:
        return float(accent_residual)
    return float(accent_residual - speaker_residual * (chance_l1 / chance_spk))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speaker identity linear probe on Whisper encoder")
    parser.add_argument("--data_root",       default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--baseline_model",  default="openai/whisper-small")
    parser.add_argument("--lora_model",      default=None)
    parser.add_argument("--split",           default="scripted",
                        choices=["scripted","spontaneous","all"])
    parser.add_argument("--output_dir",      default="results/speaker_probe")
    parser.add_argument("--layers",          default=None)
    parser.add_argument("--max_utts",        type=int, default=None)
    parser.add_argument("--n_folds",         type=int, default=5)
    # Path to accent probe results JSON (for accent-beyond-speaker computation)
    parser.add_argument("--accent_results",  default=None,
                        help="Path to accent_probe_*.json for joint analysis")
    args = parser.parse_args()

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Speaker Probe ===")
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
        max_utts=args.max_utts
    )
    print(f"      Found {len(utterances)} utterances")

    models_to_eval = {"baseline": args.baseline_model}
    if args.lora_model:
        models_to_eval["lora"] = args.lora_model

    # Load accent results if provided (for joint analysis)
    accent_results = None
    if args.accent_results:
        import json
        with open(args.accent_results) as f:
            accent_results = json.load(f)

    all_results = {}

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

        print(f"[4/4] Running speaker probes …")
        layer_results = {}

        for layer_idx in layer_indices:
            print(f"    Layer {layer_idx:2d} | probing …")
            X, _, _, speakers = records_to_arrays(records, layer_idx)
            
            #subsample if too large (to speed up probing; use fixed seed for reproducibility)
            if len(X) > 30000:
                idx = np.random.RandomState(42).choice(len(X), 30000, replace=False)
                X, speakers = X[idx], speakers[idx]

            n_speakers = len(np.unique(speakers))

            result = run_speaker_probe(X, speakers, layer_idx, args.n_folds)

            # Compute accent-beyond-speaker if accent results available
            if accent_results and model_name in accent_results:
                accent_acc = (accent_results[model_name]
                              .get(str(layer_idx), {})
                              .get("global", {})
                              .get("accuracy", float("nan")))
                if not np.isnan(accent_acc):
                    delta = compute_accent_beyond_speaker(
                        result["accuracy"], accent_acc,
                        n_speakers, NUM_L1S,
                    )
                    result["accent_beyond_speaker"] = delta
                    print(f"             accent-beyond-speaker Δ = {delta:+.3f}")

            layer_results[str(layer_idx)] = result

        all_results[model_name] = layer_results
        del base

    out_path = Path(args.output_dir) / f"speaker_probe_{args.split}.json"
    save_results(all_results, str(out_path))
    print(f"\nDone. Results → {out_path}")

    # Summary
    print("\n=== Summary: Speaker Probe Accuracy by Layer ===")
    header = f"{'Layer':>6}" + "".join(f"  {m:>14}" for m in all_results)
    print(header)
    for li in layer_indices:
        row = f"{li:>6}"
        for m in all_results:
            val = all_results[m].get(str(li), {}).get("accuracy", float("nan"))
            row += f"  {val:>14.3f}"
        print(row)


if __name__ == "__main__":
    main()

