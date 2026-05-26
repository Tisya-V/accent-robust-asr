#!/usr/bin/env python3
"""
Evaluation of E2 Latent Diffusion Bridge.

Loads bridge checkpoint, runs inference on test utterances, computes WER/MER/PER.
Results saved to CSV with same format as eval_whisper.py and eval_whisfusion.py.

Usage:
    python -m src.experiments.exp2_latent_diffusion_bridge.eval \
        --bridge_ckpt models/bridge/checkpoint_best.pt \
        --decoder whisper
"""

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
import jiwer
from tqdm import tqdm
import soundfile

from src.config import TEST_SPEAKERS, SPEAKER_L1
from src.utils.load_l2arctic import load_test_utterances
from src.utils.model_loader import load_baseline_whisper
from src.utils.bridge_utils import get_split_data_dir

import nltk

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk_data_dir = os.environ.get('NLTK_DATA')
    print(f"Downloading NLTK averaged_perceptron_tagger_eng to {nltk_data_dir}...")
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)

import g2p_en

BATCH_SIZE = 16
_G2P = g2p_en.G2p()


def text_to_phones(text: str) -> list[str]:
    """Normalised text → ARPAbet phone list (stress digits stripped)."""
    raw = _G2P(text)
    return [p.rstrip("012") for p in raw if p.strip() and p[0].isalpha()]


def utt_per(ref: str, pred: str) -> float | None:
    """G2P-derived phoneme error rate for one utterance."""
    if not ref:
        return None
    ref_phones = " ".join(text_to_phones(ref))
    pred_phones = " ".join(text_to_phones(pred))
    if not ref_phones:
        return None
    return float(jiwer.wer(ref_phones, pred_phones))


def norm(s: str) -> str:
    """Normalize text for WER computation."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_bridge_model(
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained BridgeTransformer from checkpoint, reading architecture from config.json."""
    import json
    from .model import BridgeTransformer

    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path} — needed to reconstruct model architecture")

    with open(config_path) as f:
        cfg = json.load(f)

    model = BridgeTransformer(
        d_model          = cfg.get("d_model", 256),
        n_layers         = cfg.get("n_layers", 4),
        n_heads          = cfg.get("n_heads", 8),
        dim_feedforward  = cfg.get("dim_feedforward", 1024),
        cond_acc         = cfg.get("cond_acc", False),
        parameterization = cfg.get("parameterization", "eps"),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_encoder_state(state_path: str | Path) -> torch.Tensor:
    """Load encoder state from .pt file and upcast to float32."""
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    z = state["hidden_states"]  # [1500, 768] in bfloat16 or float32
    if z.dtype == torch.bfloat16:
        z = z.float()
    return z


def bridge_inference_single(
    bridge: torch.nn.Module,
    z_acc: torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 0.5,
    device: torch.device = torch.device("cpu"),
    parameterization: str = "eps",
    speech_end: Optional[int] = None,
) -> torch.Tensor:
    """Run bridge inference on a single [1500, 768] latent."""
    from .diffusion import bridge_inference

    z_acc = z_acc.to(device=device, dtype=torch.bfloat16).unsqueeze(0)  # [1, 1500, 768]
    with torch.no_grad():
        z_nat_hat = bridge_inference(bridge, z_acc, n_steps=n_steps, sigma_max=sigma_max,
                                     parameterization=parameterization, speech_end=speech_end)
    return z_nat_hat.squeeze(0).float().cpu()  # [1500, 768]


def transcribe_with_bridge(
    utterances: list[dict],
    bridge: torch.nn.Module,
    processor,
    decoder_model,
    device: torch.device,
    decoder_type: str = "whisper",
    n_steps: int = 20,
    sigma_max: float = 0.5,
) -> list[str]:
    """
    Transcribe utterances using bridge-corrected encoder states.

    Args:
        utterances: List of test utterances with speaker, utterance_id, etc.
        bridge: Trained BridgeTransformer model
        processor: Whisper processor (for decoder input format)
        decoder_model: Decoder model (Whisper or Whisfusion)
        device: Device to run on
        decoder_type: "whisper" or "whisfusion"
        n_steps: Number of ODE steps for bridge inference
        sigma_max: Bridge noise scale — must match the value used during training

    Returns:
        List of transcriptions
    """
    from transformers.modeling_outputs import BaseModelOutput

    predictions = []
    pbar = tqdm(utterances, desc="  transcribing (bridge)", unit="utt")

    for utt in pbar:
        speaker = utt["speaker"]
        utterance_id = utt["utterance_id"]

        # Determine which split contains this utterance
        z_acc = None
        for split in ["train", "dev", "test"]:
            split_dir = get_split_data_dir(split)
            state_path = split_dir / speaker / f"{speaker}_{utterance_id}.pt"
            if not state_path.exists():
                state_path = split_dir / speaker / f"{utterance_id}.pt"

            if state_path.exists():
                z_acc = load_encoder_state(state_path)
                break

        if z_acc is None:
            print(f"[WARN] Could not find encoder state for {speaker}/{utterance_id}")
            predictions.append("")
            continue

        # Run bridge inference — stitch original L2 silence back for frames beyond speech_end
        parameterization = getattr(bridge, "parameterization", "eps")
        speech_end = utt.get("speech_end_frame")
        z_nat_hat = bridge_inference_single(bridge, z_acc, device=device,
                                            n_steps=n_steps, sigma_max=sigma_max,
                                            parameterization=parameterization,
                                            speech_end=speech_end)

        # Decode with bridge-corrected latents
        if decoder_type == "whisper":
            # Whisper decoder expects encoder_outputs as BaseModelOutput
            enc_out = BaseModelOutput(last_hidden_state=z_nat_hat.unsqueeze(0).to(device))
            with torch.no_grad():
                pred_ids = decoder_model.generate(
                    encoder_outputs=enc_out,
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                )
            pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        elif decoder_type == "whisfusion":
            raise NotImplementedError("Whisfusion decoder support not yet implemented")
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        predictions.append(pred)

    return predictions


def build_results(
    utterances: list[dict],
    predictions: list[str],
) -> pd.DataFrame:
    """Build results DataFrame matching eval_whisper.py format."""
    rows = []
    for utt, pred in zip(utterances, predictions):
        ref = norm(utt["text"])
        pred_n = norm(pred)

        word_measures = jiwer.process_words(ref, pred_n) if ref else None

        rows.append({
            "utterance_id": utt["utterance_id"],
            "speaker": utt["speaker"],
            "l1": utt["l1"],
            "domain": utt.get("domain", ""),
            "wav_path": utt["wav_path"],
            "text": utt["text"],
            "prediction": pred,
            "reference_norm": ref,
            "prediction_norm": pred_n,
            "ref_num_words": len(ref.split()),
            "utt_wer": float(word_measures.wer) if word_measures else None,
            "utt_mer": float(word_measures.mer) if word_measures else None,
            "utt_per": utt_per(ref, pred_n),
        })
    return pd.DataFrame(rows)


def evaluate_bridge(
    bridge_ckpt: str | Path,
    decoder: str = "whisper",
    output_dir: str = "results/bridge_eval",
    n_steps: int = 20,
    sigma_max: float | None = None,
    mapping_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json",
    max_utts_per_speaker: int | None = None,
    output_file: str = "bridge_predictions.csv",
) -> None:
    """
    Evaluate bridge model on test speakers.

    Args:
        bridge_ckpt: Path to checkpoint_best.pt
        decoder: "whisper" or "whisfusion"
        output_dir: Where to save results CSV
        n_steps: Number of ODE steps for bridge inference
        sigma_max: Noise schedule parameter — if None, read from config.json next to checkpoint
        mapping_path: Path to mapping_test.json (encoder states located via TEST_DATA_DIR env var)
        max_utts_per_speaker: If set, stratified subsample to this many utterances per speaker
    """
    import json as _json

    bridge_ckpt = Path(bridge_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Load bridge
    print(f"[Eval] Loading bridge from {bridge_ckpt}")
    bridge = load_bridge_model(bridge_ckpt, device)

    # Read sigma_max from config if not supplied
    if sigma_max is None:
        config_path = bridge_ckpt.parent / "config.json"
        cfg = _json.loads(config_path.read_text()) if config_path.exists() else {}
        sigma_max = cfg.get("sigma_max", 0.5)
        print(f"[Eval] sigma_max={sigma_max} (from config.json)")

    # Load decoder
    print(f"[Eval] Loading {decoder} decoder...")
    if decoder == "whisper":
        decoder_model, processor = load_baseline_whisper()
        decoder_model = decoder_model.to(device)
        decoder_model.eval()
    elif decoder == "whisfusion":
        raise NotImplementedError("Whisfusion support not yet implemented")
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    # Load test utterances from mapping_test.json — encoder states resolved via TEST_DATA_DIR
    print(f"[Eval] Loading test utterances from {mapping_path}")
    with open(mapping_path) as f:
        test_utts = _json.load(f)
    for u in test_utts:
        u.setdefault("wav_path", "")  # not needed for eval, only for CSV output

    if max_utts_per_speaker is not None:
        import random as _random
        by_speaker: dict = {}
        for u in test_utts:
            by_speaker.setdefault(u["speaker"], []).append(u)
        test_utts = []
        for spk, utts in by_speaker.items():
            _random.seed(42)
            test_utts.extend(_random.sample(utts, min(max_utts_per_speaker, len(utts))))
        print(f"  Subsampled to {max_utts_per_speaker} utts/speaker")

    print(f"  {len(test_utts)} test utterances from {len(set(u['speaker'] for u in test_utts))} speakers")

    # Run inference
    print(f"[Eval] Running bridge inference and decoding...")
    preds = transcribe_with_bridge(
        test_utts,
        bridge,
        processor,
        decoder_model,
        device,
        decoder_type=decoder,
        n_steps=n_steps,
        sigma_max=sigma_max,
    )

    # Build results
    results = build_results(test_utts, preds)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_file
    results.to_csv(out_path, index=False)

    # Print summary
    corpus_measures = jiwer.process_words(
        results["reference_norm"].tolist(),
        results["prediction_norm"].tolist(),
    )

    wer = float(corpus_measures.wer)
    mer = float(corpus_measures.mer)

    per_vals = results["utt_per"].dropna()
    per_str = f"  PER={per_vals.mean():.3f}" if len(per_vals) else "  PER=n/a"

    print(f"\n[Results]")
    print(f"  WER={wer:.3f}  MER={mer:.3f}{per_str}")
    print(f"  Saved: {out_path}")

    # Per-L1 breakdown
    print(f"\n[Per-L1 Breakdown]")
    for l1 in sorted(set(u["l1"] for u in test_utts)):
        l1_results = results[results["l1"] == l1]
        if len(l1_results) > 0:
            l1_measures = jiwer.process_words(
                l1_results["reference_norm"].tolist(),
                l1_results["prediction_norm"].tolist(),
            )
            print(f"  {l1:12s}: WER={float(l1_measures.wer):.3f}  ({len(l1_results)} utts)")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate E2 Latent Diffusion Bridge")
    parser.add_argument(
        "--bridge_ckpt",
        type=str,
        default="models/bridge/checkpoint_best.pt",
        help="Path to bridge checkpoint",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="whisper",
        choices=["whisper", "whisfusion"],
        help="Decoder to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bridge_eval",
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of ODE steps for bridge inference",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=None,
        help="Noise schedule parameter — defaults to value in config.json next to checkpoint",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_test.json",
        help="Path to mapping_test.json",
    )
    parser.add_argument(
        "--max_utts_per_speaker",
        type=int,
        default=None,
        help="Stratified subsample to this many utterances per speaker (for quick eval)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bridge_predictions.csv",
        help="Filename for the output CSV within output_dir",
    )

    args = parser.parse_args()
    evaluate_bridge(
        bridge_ckpt=args.bridge_ckpt,
        decoder=args.decoder,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        sigma_max=args.sigma_max,
        mapping_path=args.mapping_path,
        max_utts_per_speaker=args.max_utts_per_speaker,
        output_file=args.output_file,
    )
