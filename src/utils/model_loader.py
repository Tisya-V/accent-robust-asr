"""
src/utils/model_loader.py

Utility loaders for each trained model checkpoint.

Usage:
    from src.utils.model_loader import load_baseline, load_baseline_lora, load_ctc_aux

    model, processor = load_baseline()
    model, processor = load_baseline_lora()
    model, processor = load_ctc_aux()        # merged LoRA, no aux heads
    model, processor = load_ctc_aux_full()   # WhisperWithAuxHeads (with heads intact)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.training.whisper_aux import WhisperWithAuxHeads

BASE_MODEL_ID   = "openai/whisper-small"
BASELINE_LORA   = "models/baseline_loraft"
CTC_AUX_DIR     = "models/ctc_aux"


def _processor(model_id: str = BASE_MODEL_ID) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_id, local_files_only=True)


def get_model_registry(device):
    return {
        "baseline":      {
            "label": "Zero-shot",    
            "loader": lambda: load_baseline(device=device)
        },

        "baseline_lora": {
            "label": "Naive LoRA FT",      
            "loader": lambda: load_baseline_lora(device=device)
        },

        "baseline_lora_heldout_chinese": {
            "label": "Naive LoRA FT [held-out Chinese]",      
            "loader": lambda: load_baseline_lora(device=device, checkpoint_dir="models/baseline_loraft_heldout_chinese")
        },

        "ctc_aux": {
            "label": "CTC Aux",      
            "loader": lambda: load_ctc_aux(device=device)
        },

        "no_aux": {
            "label": "No Aux",
            "loader": lambda: load_ctc_aux(device=device, checkpoint_dir="models/no_aux")
        },
    }


# ---------------------------------------------------------------------------
# Baseline — frozen openai/whisper-small, no fine-tuning
# ---------------------------------------------------------------------------

def load_baseline(
    device: str | None = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Plain whisper-small, no fine-tuning.
    Use as the zero-shot reference point.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, local_files_only=True
    ).to(device)
    model.eval()
    processor = _processor()
    return model, processor


# ---------------------------------------------------------------------------
# Baseline LoRA — whisper-small fine-tuned with LoRA, no aux heads
# ---------------------------------------------------------------------------

def load_baseline_lora(
    checkpoint_dir: str = BASELINE_LORA,
    device: str | None = None,
    merged: bool = True,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    LoRA fine-tuned whisper-small (standard ASR objective only).

    Args:
        checkpoint_dir: path to saved PEFT checkpoint
        merged:         if True, merge LoRA weights and return a plain
                        WhisperForConditionalGeneration (faster inference).
                        if False, return PeftModel (useful for inspecting adapters).
    """
    from peft import PeftModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)

    if merged:
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()
    processor = _processor(checkpoint_dir)
    return model, processor


# ---------------------------------------------------------------------------
# CTC Aux — whisper-small + LoRA fine-tuned with CTC phoneme auxiliary loss
# ---------------------------------------------------------------------------

def load_ctc_aux(
    checkpoint_dir: str = CTC_AUX_DIR,
    device: str | None = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    CTC-aux fine-tuned model, returned as a plain WhisperForConditionalGeneration
    (LoRA merged, aux heads discarded).

    Use this for WER evaluation and probing — identical interface to load_baseline()
    and load_baseline_lora().
    """
    from peft import PeftModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(checkpoint_dir) / "best"

    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, str(ckpt))
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    # Processor saved alongside checkpoint
    processor_path = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else BASE_MODEL_ID
    processor = _processor(processor_path)
    return model, processor


def load_ctc_aux_full(
    checkpoint_dir: str = CTC_AUX_DIR,
    device: str | None = None,
    lambda_ctc: float = 0.3,
    lambda_feat: float = 0.0,
) -> Tuple[WhisperWithAuxHeads, WhisperProcessor]:  # noqa: F821
    """
    CTC-aux model with aux heads intact (WhisperWithAuxHeads wrapper).

    Use this if you need the CTC head outputs directly (e.g. phoneme forced
    alignment, or continuing training).

    Args:
        lambda_ctc / lambda_feat: must match the values used during training
                                  so hooks are registered correctly.
    """
    from peft import PeftModel
    from src.training.whisper_aux import WhisperWithAuxHeads

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(checkpoint_dir) / "best"
    ctc_head_path  = Path(checkpoint_dir) / "best_ctc_head.pt"
    feat_head_path = Path(checkpoint_dir) / "best_feat_head.pt"

    model = WhisperWithAuxHeads(
        model_name  = BASE_MODEL_ID,
        lambda_ctc  = lambda_ctc,
        lambda_feat = lambda_feat,
    )

    # Load LoRA weights into whisper backbone
    model.whisper = PeftModel.from_pretrained(model.whisper, str(ckpt))
    model.whisper = model.whisper.merge_and_unload()
    model._register_hooks()  # re-register hooks to capture from merged model

    # Load aux head weights
    if ctc_head_path.exists():
        model.ctc_head.load_state_dict(
            torch.load(ctc_head_path, map_location=device)
        )
    if feat_head_path.exists():
        model.feat_head.load_state_dict(
            torch.load(feat_head_path, map_location=device)
        )

    model = model.to(device)
    model.eval()

    processor_path = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else BASE_MODEL_ID
    processor = _processor(processor_path)
    return model, processor


# ---------------------------------------------------------------------------
# Convenience: load all models at once (for probing / eval scripts)
# ---------------------------------------------------------------------------

def load_all_for_probing(
    device: str | None = None,
) -> dict:
    """
    Returns a dict of {name: (model, processor)} for all available checkpoints.
    Models are plain WhisperForConditionalGeneration (merged, eval mode).

    Example:
        models = load_all_for_probing()
        for name, (model, proc) in models.items():
            records = build_embedding_dataset(model, proc, utterances, ...)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = {}

    out["baseline"] = load_baseline(device=device)

    if Path(BASELINE_LORA).exists():
        out["baseline_lora"] = load_baseline_lora(device=device)
    else:
        print(f"[WARN] {BASELINE_LORA} not found, skipping")

    if (Path(CTC_AUX_DIR) / "best").exists():
        out["ctc_aux"] = load_ctc_aux(device=device)
    else:
        print(f"[WARN] {CTC_AUX_DIR}/best not found, skipping")

    print(f"Loaded models: {list(out.keys())}")
    return out