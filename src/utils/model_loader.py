"""
src/utils/model_loader.py

Utility loaders for each trained model checkpoint.

Usage:
    from src.utils.model_loader import load_baseline_whisper, load_finetuned_whisper

    model, processor = load_baseline()
    model, processor = load_finetuned_whisper()
    model, processor = load_finetuned_whisper(checkpoint_dir=WHISPER_FT_HOC_DIR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


BASE_MODEL_ID   = "openai/whisper-small"
WHISPER_FT_DIR = "models/whisper_ft"
WHISPER_FT_HOC_DIR = "models/whisper_ft_hoc"


def _processor(model_id: str = BASE_MODEL_ID) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_id, local_files_only=True)


def get_model_registry(device):
    return {
        "baseline":      {
            "label": "Zero-shot",    
            "loader": lambda: load_baseline_whisper(device=device)
        },
        "whisper_ft": {
            "label": "Whisper Fine-tuned",
            "loader": lambda: load_finetuned_whisper(device=device, checkpoint_dir=WHISPER_FT_DIR)
        },
        "whisper_ft_hoc": {
            "label": "Whisper Fine-tuned [Held-out Chinese]",
            "loader": lambda: load_finetuned_whisper(device=device, checkpoint_dir=WHISPER_FT_HOC_DIR)
        }
    }


# ---------------------------------------------------------------------------
# Baseline — frozen openai/whisper-small, no fine-tuning
# ---------------------------------------------------------------------------

def load_baseline_whisper(
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


def load_finetuned_whisper(
    checkpoint_dir: str = WHISPER_FT_DIR,
    device: str | None = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Fine-tuned model, returned as a plain WhisperForConditionalGeneration
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