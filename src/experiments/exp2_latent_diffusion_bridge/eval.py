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

BATCH_SIZE = 16              # Whisper decode batch size (Phase 2 -- full 1500-frame sequences)
BRIDGE_BATCH_SIZE = 64       # Bridge ODE batch size (Phase 1 -- ~1.4GB activations on A30 24GB
                             # at d_model=768; utterances are sorted by inf_len before chunking
                             # so each batch's shared crop length stays close to every member's own)
_G2P = g2p_en.G2p()


def text_to_phones(text: str) -> list[str]:
    """Normalised text → ARPAbet phone list (stress digits stripped)."""
    text = re.sub(r"\d+", "", text)  # inflect raises NumOutOfRangeError on large numbers
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

    # Bake EMA-smoothed weights into the model permanently — eval should run on the
    # same weights val_epoch judged "best" during training, not the noisier raw
    # live weights. Pre-EMA checkpoints (no "ema_state_dict") fall through unchanged.
    if "ema_state_dict" in ckpt:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.get("ema_decay", 0.999))
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.copy_to(model.parameters())
        print(f"[Eval] Baked EMA-averaged weights into model from {ckpt_path}")

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


def load_tnat_predictor(
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load TNatPredictor from checkpoint, reading config from sibling config.json."""
    import json
    from .train_tnat_predictor import TNatPredictor

    ckpt_path = Path(ckpt_path)
    cfg_path  = ckpt_path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    predictor = TNatPredictor(
        pool_dim=cfg.get("pool_dim", 768),
        hidden=cfg.get("hidden", [256, 64]),
    )
    predictor.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    predictor = predictor.to(device).eval()
    return predictor


def _predict_t_nat(
    T_l2: int,
    z_acc_dev: torch.Tensor,
    predictor: Optional[torch.nn.Module],
    tnat_buffer: int,
    device: torch.device,
) -> int:
    """Estimate T_nat from pooled accented features via `predictor`; T_l2 if no predictor."""
    if predictor is None:
        return T_l2
    from .train_tnat_predictor import T_NORM
    with torch.no_grad():
        pool   = z_acc_dev[:T_l2].mean(dim=0, keepdim=True)
        t_l2_n = torch.tensor([[T_l2 / T_NORM]], device=device)
        T_nat  = int(round(predictor(pool, t_l2_n.squeeze(-1)).item() * T_NORM))
    return max(1, min(T_nat + tnat_buffer, 1500))


def _resolve_inference_lengths(
    alignment: str,
    T_l2: int,
    z_acc_dev: torch.Tensor,
    predictor: Optional[torch.nn.Module],
    tnat_buffer: int,
    device: torch.device,
) -> tuple[int, int]:
    """Resolve the (T_l2, T_nat) pair to pass into `bridge_inference` for `alignment`.

    - dtw_fixed: no N(t) morphing, runs entirely in L2 frame space — T_nat := T_l2,
      T_nat prediction is meaningless and skipped.
    - position: no N(t) morphing either — identity correspondence has no second
      alignment to morph between, so the active region is fixed at the constant
      max(T_l2, predicted T_nat); both endpoints collapse to that span, mirroring
      dtw_fixed's collapse to a constant.
    - dtw: N(t) genuinely morphs between T_l2 and predicted T_nat — pass both through
      unchanged.
    """
    if alignment == "dtw_fixed":
        return T_l2, T_l2

    T_nat = _predict_t_nat(T_l2, z_acc_dev, predictor, tnat_buffer, device)

    if alignment == "position":
        span = max(T_l2, T_nat)
        return span, span

    return T_l2, T_nat


def transcribe_with_bridge(
    utterances: list[dict],
    bridge: torch.nn.Module,
    processor,
    decoder_model,
    device: torch.device,
    decoder_type: str = "whisper",
    n_steps: int = 20,
    sigma_max: float = 0.5,
    alignment: str = "dtw",
    predictor: Optional[torch.nn.Module] = None,
    tnat_buffer: int = 0,
) -> list[str]:
    """
    Transcribe utterances using bridge-corrected encoder states.

    Phase 1: bridge ODE inference one utterance at a time (sequential by design).
    Phase 2: batch Whisper decoding over accumulated z_nat_hat tensors.
    """
    from transformers.modeling_outputs import BaseModelOutput

    if decoder_type not in ("whisper",):
        raise NotImplementedError(f"Batched decoding not implemented for {decoder_type}")

    parameterization = getattr(bridge, "parameterization", "eps")

    from .diffusion import bridge_inference

    # --- Phase 0: resolve (state_path, T_l2, T_nat, inf_len) per utterance ---
    # Store only the resolved path + lengths here, NOT the loaded [1500, 768] float32
    # tensor (~4.4MB each -- ~34GB across the full test set). Holding all of them
    # simultaneously (to enable sorting before batching) on top of the z_nat_hats
    # accumulation in Phase 1 (which grows to a similar ~34GB) is what pushed peak
    # host memory past the SLURM --mem=64G ceiling and triggered an oom_kill at
    # batch 106/122. z_acc is reloaded on demand, per batch, in Phase 1 below --
    # restoring the streaming memory profile the original sequential code had, at
    # the cost of reading each .pt file from disk twice (cheap: Phase 0 alone reads
    # all 7796 files in ~8 minutes).
    info: list[tuple[Path, int, int, int] | None] = []  # (state_path, T_l2, T_nat, inf_len) or None if skipped

    for utt in tqdm(utterances, desc="  resolving lengths", unit="utt"):
        speaker      = utt["speaker"]
        utterance_id = utt["utterance_id"]

        z_acc = None
        state_path = None
        for split in ["train", "dev", "test"]:
            split_dir = get_split_data_dir(split)
            candidate = split_dir / speaker / f"{speaker}_{utterance_id}.pt"
            if not candidate.exists():
                candidate = split_dir / speaker / f"{utterance_id}.pt"
            if candidate.exists():
                state_path = candidate
                z_acc = load_encoder_state(state_path)
                break

        if z_acc is None:
            print(f"[WARN] Could not find encoder state for {speaker}/{utterance_id}")
            info.append(None)
            continue

        speech_end = utt.get("speech_end_frame")
        if speech_end is None:
            print(f"[WARN] No speech_end_frame for {utt['speaker']}/{utt['utterance_id']} — skipping")
            info.append(None)
            continue

        z_acc_dev   = z_acc.to(device=device, dtype=torch.float32)
        T_l2, T_nat = _resolve_inference_lengths(alignment, speech_end, z_acc_dev, predictor, tnat_buffer, device)
        info.append((state_path, T_l2, T_nat, max(T_l2, T_nat) + 1))

    # --- Phase 1: batched bridge ODE inference, sorted by inf_len so each batch's
    # shared crop length stays close to every member's own (minimizes padding waste).
    # z_acc is reloaded from disk per chunk, in sorted order -- see Phase 0 comment. ---
    valid_idx = [i for i, x in enumerate(info) if x is not None]
    order     = sorted(valid_idx, key=lambda i: info[i][3])

    z_nat_hats: list[torch.Tensor | None] = [None] * len(utterances)

    for chunk_start in tqdm(range(0, len(order), BRIDGE_BATCH_SIZE),
                            desc="  bridge inference", unit="batch"):
        chunk = order[chunk_start : chunk_start + BRIDGE_BATCH_SIZE]

        z_acc_batch = torch.stack([load_encoder_state(info[i][0]) for i in chunk]).to(device=device, dtype=torch.bfloat16)  # [B, 1500, 768]
        T_l2_batch  = torch.tensor([info[i][1] for i in chunk], device=device, dtype=torch.long)
        T_nat_batch = torch.tensor([info[i][2] for i in chunk], device=device, dtype=torch.long)
        inf_lens    = torch.tensor([info[i][3] for i in chunk], device=device, dtype=torch.long)
        L_buf       = z_acc_batch.shape[1]
        kpm         = (torch.arange(L_buf, device=device).unsqueeze(0) < inf_lens.unsqueeze(1))  # [B, L] True=real

        with torch.no_grad():
            z_hat_batch = bridge_inference(
                bridge, z_acc_batch, T_l2=T_l2_batch, T_nat=T_nat_batch,
                n_steps=n_steps, sigma_max=sigma_max, parameterization=parameterization,
                key_padding_mask=kpm,
            )

        z_hat_batch = z_hat_batch.float().cpu()
        for j, i in enumerate(chunk):
            z_nat_hats[i] = z_hat_batch[j]  # [1500, 768] float32 CPU

    # --- Phase 2: batched Whisper decode ---
    # All z_nat_hat are [1500, 768] — no padding needed.
    valid_idx  = [i for i, z in enumerate(z_nat_hats) if z is not None]
    predictions = [""] * len(utterances)

    for batch_start in tqdm(range(0, len(valid_idx), BATCH_SIZE),
                            desc="  whisper decode", unit="batch"):
        batch_idx  = valid_idx[batch_start : batch_start + BATCH_SIZE]
        batch_tens  = torch.stack([z_nat_hats[i] for i in batch_idx]).to(device)  # [B, 1500, 768]
        enc_out     = BaseModelOutput(last_hidden_state=batch_tens)
        # Whisper's _maybe_reduce_batch needs input_features to resize the batch when shorter
        # sequences finish before longer ones. Provide a dummy — encoder_outputs drives the forward pass.
        dummy_feats = torch.zeros(len(batch_idx), 80, 3000, device=device, dtype=batch_tens.dtype)
        with torch.no_grad():
            pred_ids = decoder_model.generate(
                input_features=dummy_feats,
                encoder_outputs=enc_out,
                language="en",
                task="transcribe",
                temperature=0.0,
            )
        batch_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for i, pred in zip(batch_idx, batch_preds):
            predictions[i] = pred

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
    predictor_ckpt: str | Path | None = None,
    tnat_buffer: int = 0,
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

    predictor = None
    if predictor_ckpt is not None:
        print(f"[Eval] Loading T_nat predictor from {predictor_ckpt}")
        predictor = load_tnat_predictor(predictor_ckpt, device)
        print("[Eval] N(t) mask inference enabled")

    # Read sigma_max and alignment from config if not supplied
    config_path = bridge_ckpt.parent / "config.json"
    cfg = _json.loads(config_path.read_text()) if config_path.exists() else {}
    if sigma_max is None:
        sigma_max = cfg.get("sigma_max", 0.5)
        print(f"[Eval] sigma_max={sigma_max} (from config.json)")
    alignment = cfg.get("alignment", "dtw")
    print(f"[Eval] alignment={alignment} (from config.json)")

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

    # BDL (English native) appears as both l2_test and native_sanity_check — drop the l2_test duplicate
    # before subsampling so the full native_sanity_check pool is available for sampling.
    test_utts = [u for u in test_utts if not (u.get("eval_type") == "l2_test" and u.get("l1") == "English")]

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
        alignment=alignment,
        predictor=predictor,
        tnat_buffer=tnat_buffer,
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
    per_l1 = {}
    for l1 in sorted(set(u["l1"] for u in test_utts)):
        l1_results = results[results["l1"] == l1]
        if len(l1_results) > 0:
            l1_measures = jiwer.process_words(
                l1_results["reference_norm"].tolist(),
                l1_results["prediction_norm"].tolist(),
            )
            l1_wer = float(l1_measures.wer)
            per_l1[l1] = {"wer": l1_wer, "n_utts": len(l1_results)}
            print(f"  {l1:12s}: WER={l1_wer:.3f}  ({len(l1_results)} utts)")

    # Save the summary (results + breakdown) next to config.json/history.json so the
    # model's setup and its eval outcome can be checked together in one place.
    # n_test_utts + max_utts_per_speaker reveal whether this ran on a subsample.
    summary = {
        "wer": wer,
        "mer": mer,
        "per": float(per_vals.mean()) if len(per_vals) else None,
        "n_test_utts": len(test_utts),
        "max_utts_per_speaker": max_utts_per_speaker,
        "per_l1": per_l1,
        "predictions_csv": str(out_path),
    }
    summary_path = bridge_ckpt.parent / "eval_summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2))
    print(f"  Saved summary: {summary_path}")

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
    parser.add_argument(
        "--predictor_ckpt",
        type=str,
        default=None,
        help="Path to TNatPredictor checkpoint (enables DTW N(t) mask at inference)",
    )
    parser.add_argument(
        "--tnat_buffer",
        type=int,
        default=0,
        help="Extra frames added to T_nat_hat before bridge inference to guard against truncation",
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
        predictor_ckpt=args.predictor_ckpt,
        tnat_buffer=args.tnat_buffer,
    )
