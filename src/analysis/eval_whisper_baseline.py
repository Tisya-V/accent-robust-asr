#!/usr/bin/env python3
"""
Baseline Whisper evaluation over all encoder states (no bridge).

Sweeps every unique .pt file referenced by mapping_train_v2.json,
mapping_dev_v2.json, and mapping_test.json.  Passes each encoder state
directly to the Whisper decoder (bypassing the bridge) and records WER/MER/PER.

Covers both L2 (z_acc) and native (z_nat) encoder states, giving:
  - L2  → unassisted Whisper WER on accented speech   (bridge lower bound)
  - nat → Whisper WER on native speech                 (upper bound / oracle)

Output
------
results/whisper_baseline/whisper_baseline.csv
    speaker, utterance_id, l1, speaker_type, bridge_split,
    text, whisper_pred, reference_norm, prediction_norm,
    wer, mer, per

Usage
-----
  source scripts/slurm_env.sh
  python -m src.experiments.exp2_latent_diffusion_bridge.eval_whisper_baseline \
      [--output_dir results/whisper_baseline] \
      [--batch_size 32]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import jiwer
import pandas as pd
import torch
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

# Reuse helpers from the existing bridge eval
from src.experiments.exp2_latent_diffusion_bridge.eval import (
    norm,
    utt_per,
    load_encoder_state,
)
from src.utils.model_loader import load_baseline_whisper
from src.utils.bridge_utils import get_split_data_dir


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

DATA_DIR = Path("src/experiments/exp2_latent_diffusion_bridge/data")

MAPPING_FILES = {
    "train": DATA_DIR / "mapping_train_v2.json",
    "dev":   DATA_DIR / "mapping_dev_v2.json",
    "test":  DATA_DIR / "mapping_test.json",
}


def _find_pt(rel_path: str) -> Path | None:
    """Search train/dev/test processed data dirs for a relative encoder state path."""
    for split_name in ["train", "dev", "test"]:
        p = get_split_data_dir(split_name) / rel_path
        if p.exists():
            return p
    return None


def collect_entries() -> list[dict]:
    """
    Build a deduplicated list of entries to evaluate.

    Each entry:
        encoder_state_path  – relative path used as the dedup key
        abs_path            – resolved absolute Path to the .pt file
        speaker             – speaker ID
        utterance_id        – arctic_XXXXX
        l1                  – L1 background (or "English" for native)
        speaker_type        – "l2" or "native"
        bridge_split        – "train" | "dev" | "test"
        text                – reference transcript
    """
    seen: set[str] = set()
    entries: list[dict] = []

    for split_name, mapping_path in MAPPING_FILES.items():
        if not mapping_path.exists():
            print(f"[WARN] mapping not found: {mapping_path}, skipping")
            continue

        with open(mapping_path) as f:
            pairs = json.load(f)

        for pair in tqdm(pairs, desc=f"Collecting entries from {mapping_path.name}"):
            # Each mapping entry contributes up to two encoder states:
            # z_acc (L2 speaker) and z_nat (native speaker).
            # mapping_test.json only has L2 speaker metadata.

            candidates = []

            # L2 encoder state
            l2_rel = pair.get("l2_encoder_state_path")
            if l2_rel:
                candidates.append({
                    "encoder_state_path": l2_rel,
                    "speaker":            pair.get("l2_speaker", pair.get("speaker", "")),
                    "utterance_id":       pair.get("l2_utterance_id", pair.get("utterance_id", "")),
                    "l1":                 pair.get("l1", "Unknown"),
                    "speaker_type":       "l2",
                    "bridge_split":       split_name,
                    "text":               pair.get("text", ""),
                })

            # Native encoder state (absent in mapping_test.json)
            nat_rel = pair.get("nat_encoder_state_path")
            if nat_rel:
                candidates.append({
                    "encoder_state_path": nat_rel,
                    "speaker":            pair.get("nat_speaker", ""),
                    "utterance_id":       pair.get("nat_utterance_id",
                                                   pair.get("utterance_id", "")),
                    "l1":                 "English",
                    "speaker_type":       "native",
                    "bridge_split":       split_name,
                    "text":               pair.get("text", ""),
                })

            # mapping_test.json entries (no l2_encoder_state_path field)
            if not l2_rel and not nat_rel:
                spk = pair.get("speaker", "")
                utt = pair.get("utterance_id", "")
                rel = f"{spk}/{spk}_{utt}.pt"
                eval_type = pair.get("eval_type", "l2_test")
                # BDL appears as both l2_test and native_sanity_check in mapping_test.json.
                # Skip the l2_test duplicate — it is captured correctly as native_sanity_check.
                if eval_type == "l2_test" and pair.get("l1") == "English":
                    continue
                candidates.append({
                    "encoder_state_path": rel,
                    "speaker":            spk,
                    "utterance_id":       utt,
                    "l1":                 pair.get("l1", "Unknown"),
                    "speaker_type":       "native" if "native" in eval_type else "l2",
                    "bridge_split":       split_name,
                    "text":               pair.get("text", ""),
                })

            for cand in candidates:
                rel = cand["encoder_state_path"]
                if rel in seen:
                    continue
                abs_p = _find_pt(rel)
                if abs_p is None:
                    continue
                cand["abs_path"] = abs_p
                seen.add(rel)
                entries.append(cand)

    return entries


# ---------------------------------------------------------------------------
# Whisper batch decode
# ---------------------------------------------------------------------------

def batch_decode(
    whisper_model,
    processor,
    states: list[torch.Tensor],
    device: torch.device,
) -> list[str]:
    """Decode a list of [1500, 768] encoder state tensors with Whisper."""
    stacked = torch.stack(states).to(device)  # [B, 1500, 768] float32
    enc_out = BaseModelOutput(last_hidden_state=stacked)
    with torch.no_grad():
        ids = whisper_model.generate(
            encoder_outputs=enc_out,
            language="en",
            task="transcribe",
            temperature=0.0,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: str = "results/whisper_baseline", batch_size: int = 16) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[eval_whisper_baseline] Collecting entries...")
    from src.utils.bridge_utils import get_split_data_dir
    for split in ("train", "dev", "test"):
        d = get_split_data_dir(split)
        print(f"  {split} data dir: {d}  (exists={d.exists()})")
    entries = collect_entries()
    print(f"  {len(entries)} unique encoder states to evaluate")
    if not entries:
        print("[ERROR] No entries collected — check that TRAIN_DATA_DIR / DEV_DATA_DIR / TEST_DATA_DIR")
        print("        env vars point to the processed data dirs (source scripts/slurm_env.sh).")
        return

    l2_count  = sum(1 for e in entries if e["speaker_type"] == "l2")
    nat_count = sum(1 for e in entries if e["speaker_type"] == "native")
    print(f"  L2: {l2_count}  native: {nat_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_whisper_baseline] Device: {device}")
    print("[eval_whisper_baseline] Loading Whisper...")
    whisper_model, processor = load_baseline_whisper()
    whisper_model = whisper_model.to(device).eval()

    rows = []
    batch_entries: list[dict] = []
    batch_states:  list[torch.Tensor] = []

    def flush():
        if not batch_states:
            return
        preds = batch_decode(whisper_model, processor, batch_states, device)
        for entry, pred in zip(batch_entries, preds):
            ref    = norm(entry["text"])
            pred_n = norm(pred)
            wm     = jiwer.process_words(ref, pred_n) if ref else None
            rows.append({
                "speaker":          entry["speaker"],
                "utterance_id":     entry["utterance_id"],
                "l1":               entry["l1"],
                "speaker_type":     entry["speaker_type"],
                "bridge_split":     entry["bridge_split"],
                "text":             entry["text"],
                "whisper_pred":     pred,
                "reference_norm":   ref,
                "prediction_norm":  pred_n,
                "wer": float(wm.wer) if wm else None,
                "mer": float(wm.mer) if wm else None,
                "per": utt_per(ref, pred_n),
            })
        batch_entries.clear()
        batch_states.clear()

    for entry in tqdm(entries, desc="Evaluating"):
        try:
            z = load_encoder_state(entry["abs_path"])  # [1500, 768] float32
        except Exception as e:
            print(f"[WARN] Failed to load {entry['abs_path']}: {e}")
            continue

        batch_entries.append(entry)
        batch_states.append(z)

        if len(batch_states) >= batch_size:
            flush()

    flush()  # remaining

    df = pd.DataFrame(rows)
    out_path = out_dir / "whisper_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[eval_whisper_baseline] Saved {len(df)} rows → {out_path}")

    if df.empty:
        print("[WARN] No rows to summarise.")
        return

    # Summary
    print("\n=== WER by speaker_type ===")
    print(df.groupby("speaker_type")["wer"].agg(["mean", "median", "count"]).round(4))

    print("\n=== WER by L1 ===")
    print(df.groupby("l1")["wer"].agg(["mean", "count"]).round(4))

    print("\n=== L2 WER by bridge_split ===")
    l2 = df[df["speaker_type"] == "l2"]
    print(l2.groupby("bridge_split")["wer"].agg(["mean", "count"]).round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline Whisper evaluation over all encoder states (no bridge)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/whisper_baseline",
        help="Directory to write whisper_baseline.csv",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Whisper decode batch size",
    )
    args = parser.parse_args()
    run(output_dir=args.output_dir, batch_size=args.batch_size)
