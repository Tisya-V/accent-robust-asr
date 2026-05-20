#!/usr/bin/env python3
"""
Evaluate the position-based latent corrector on test speakers.

Loads best checkpoint, runs model(z_acc) -> z_nat_hat, decodes via
Whisper decoder, reports WER/PER per-speaker and per-L1.

Usage:
    python -m src.experiments.exp2_transformer_bridge.eval \
        --ckpt models/corrector_position/checkpoint_best.pt \
        --out_dir results/corrector_position_eval
"""

from __future__ import annotations

import argparse
import re
import os
import json
from pathlib import Path

import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from src.utils.model_loader import load_baseline_whisper
from src.utils.bridge_utils import get_split_data_dir
from .model import LatentCorrectorTransformer

import nltk
import g2p_en

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', download_dir=os.environ.get('NLTK_DATA'))

_G2P = g2p_en.G2p()


def text_to_phones(text: str) -> list[str]:
    raw = _G2P(text)
    return [p.rstrip("012") for p in raw if p.strip() and p[0].isalpha()]


def utt_per(ref: str, pred: str) -> float | None:
    if not ref:
        return None
    ref_phones = " ".join(text_to_phones(ref))
    pred_phones = " ".join(text_to_phones(pred))
    return float(jiwer.wer(ref_phones, pred_phones)) if ref_phones else None


def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_corrector(ckpt_path: str | Path, device: torch.device) -> LatentCorrectorTransformer:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Read model config from config.json saved alongside checkpoint
    config_path = ckpt_path.parent / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    model = LatentCorrectorTransformer(
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 1024),
    )
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def load_encoder_state(state_path: Path) -> torch.Tensor:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    z = state["hidden_states"]  # [1500, 768]
    return z.float() if z.dtype == torch.bfloat16 else z


def find_encoder_state(speaker: str, utterance_id: str) -> Path | None:
    for split in ["train", "dev", "test"]:
        split_dir = get_split_data_dir(split)
        for name in [f"{speaker}_{utterance_id}.pt", f"{utterance_id}.pt"]:
            p = split_dir / speaker / name
            if p.exists():
                return p
    return None


def evaluate(
    ckpt: str,
    out_dir: str = "results/corrector_position_eval",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    print(f"[Eval] Loading corrector from {ckpt}")
    model = load_corrector(ckpt, device)
    print(f"[Eval] Params: {sum(p.numel() for p in model.parameters()):,}")

    print("[Eval] Loading Whisper decoder...")
    whisper_model, processor = load_baseline_whisper()
    whisper_model = whisper_model.to(device).eval()

    print("[Eval] Loading test utterances from mapping_test.json...")
    mapping_test_path = Path("src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json")
    with open(mapping_test_path) as f:
        all_test = json.load(f)
    # L2 speakers from l2_test; BDL from native_sanity_check only (avoids double-counting)
    test_utts = [u for u in all_test if
                 (u["eval_type"] == "l2_test" and u["l1"] != "English") or
                 u["eval_type"] == "native_sanity_check"]
    print(f"  {len(test_utts)} utterances from {len(set(u['speaker'] for u in test_utts))} speakers")

    rows = []
    for utt in tqdm(test_utts, desc="Evaluating"):
        speaker = utt["speaker"]
        utt_id = utt["utterance_id"]

        state_path = find_encoder_state(speaker, utt_id)
        if state_path is None:
            print(f"[WARN] No encoder state for {speaker}/{utt_id}")
            continue

        z_acc = load_encoder_state(state_path).to(device).unsqueeze(0)  # [1, 1500, 768]

        with torch.no_grad():
            z_nat_hat = model(z_acc)  # [1, 1500, 768] — no speech_end needed at inference
            enc_out = BaseModelOutput(last_hidden_state=z_nat_hat)
            pred_ids = whisper_model.generate(
                encoder_outputs=enc_out,
                language="en",
                task="transcribe",
                temperature=0.0,
            )
        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        ref = norm(utt["text"])
        pred_n = norm(pred)
        word_measures = jiwer.process_words(ref, pred_n) if ref else None

        rows.append({
            "utterance_id": utt_id,
            "speaker": speaker,
            "l1": utt["l1"],
            "text": utt["text"],
            "prediction": pred,
            "reference_norm": ref,
            "prediction_norm": pred_n,
            "utt_wer": float(word_measures.wer) if word_measures else None,
            "utt_per": utt_per(ref, pred_n),
        })

    results = pd.DataFrame(rows)
    if results.empty:
        print("[Eval] No results — check encoder state paths.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "predictions.csv", index=False)

    corpus = jiwer.process_words(results["reference_norm"].tolist(), results["prediction_norm"].tolist())
    print(f"\n[Results] WER={float(corpus.wer):.3f}  MER={float(corpus.mer):.3f}")
    print("[Per-L1]")
    for l1 in sorted(results["l1"].unique()):
        sub = results[results["l1"] == l1]
        m = jiwer.process_words(sub["reference_norm"].tolist(), sub["prediction_norm"].tolist())
        print(f"  {l1:12s}: WER={float(m.wer):.3f}  ({len(sub)} utts)")
    print(f"\nSaved: {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/corrector_position/checkpoint_best.pt")
    parser.add_argument("--out_dir", default="results/corrector_position_eval")
    args = parser.parse_args()
    evaluate(**vars(args))
