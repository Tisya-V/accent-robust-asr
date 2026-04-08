"""
naive_lora_ft.py
Fine-tunes whisper-small on L2-ARCTIC (local files) with LoRA.
No auxiliary objectives — pure ASR fine-tuning as the baseline_lora model.

Usage:
    python naive_lora_ft.py
    python naive_lora_ft.py --output_dir models/baseline_lora --epochs 5 --lora_r 8
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import librosa
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.config import LOCAL_L2ARCTIC_DIR, MODEL_ID
from src.utils.load_l2arctic import load_train_dev_utterances, load_test_utterances
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Dataset ────────────────────────────────────────────────────────────────────


class L2ArcticDataset(Dataset):
    """Loads utterances from local L2-ARCTIC wav + transcript pairs."""

    def __init__(self, utterances: List[Dict], processor: WhisperProcessor):
        self.utterances = utterances
        self.processor  = processor

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt = self.utterances[idx]

        audio, sr = sf.read(utt["wav_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer(
            utt["text"], return_tensors="pt"
        ).input_ids[0]

        return {
            "input_features":  inputs.input_features[0],
            "attention_mask":  inputs.attention_mask[0],
            "labels":          labels,
        }


# ── Collator ───────────────────────────────────────────────────────────────────


@dataclass
class WhisperDataCollator:
    processor:    WhisperProcessor
    pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features  = torch.stack([f["input_features"] for f in features])
        attention_masks = torch.stack([f["attention_mask"]  for f in features])
        label_list      = [f["labels"] for f in features]
        max_len         = max(l.shape[0] for l in label_list)
        padded_labels   = torch.full(
            (len(label_list), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, lab in enumerate(label_list):
            padded_labels[i, : lab.shape[0]] = lab
        return {
            "input_features": input_features,
            "attention_mask":  attention_masks,
            "labels":          padded_labels,
        }


# ── Model ──────────────────────────────────────────────────────────────────────


def build_lora_model(
    model_id:   str,
    r:          int   = 8,
    lora_alpha: int   = 16,
    dropout:    float = 0.05,
) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    config = LoraConfig(
        r              = r,
        lora_alpha     = lora_alpha,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = dropout,
        bias           = "none",
        inference_mode = False,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_loss_curve(log_history: list[dict], out_path: Path) -> None:
    """Plot train loss + dev loss/WER from Seq2SeqTrainer log_history."""
    train_steps, train_loss         = [], []
    dev_epochs, dev_loss, dev_wer   = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            dev_epochs.append(entry.get("epoch", entry["step"]))
            dev_loss.append(entry["eval_loss"])
            if "eval_wer" in entry:
                dev_wer.append(entry["eval_wer"])

    has_wer = len(dev_wer) == len(dev_epochs)
    fig, axes = plt.subplots(1, 2 if has_wer else 1,
                             figsize=(10 if has_wer else 5, 4), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    ax.plot(train_steps, train_loss, color="#2196F3", alpha=0.4, label="train (step)")
    ax2 = ax.twinx()
    ax2.plot(dev_epochs, dev_loss, "s--", color="#FF5722", label="dev loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Train loss"); ax2.set_ylabel("Dev loss")
    ax.set_title("Loss")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(alpha=0.3)

    if has_wer:
        ax = axes[1]
        ax.plot(dev_epochs, dev_wer, "s-", color="#4CAF50")
        ax.set_xlabel("Epoch"); ax.set_ylabel("WER")
        ax.set_title("Dev WER"); ax.grid(alpha=0.3)

    fig.suptitle("baseline_lora training curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve → {out_path}")


# ── Training ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Naive LoRA fine-tuning on L2-ARCTIC")
    parser.add_argument("--output_dir", default="models/baseline_lora")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lora_r",     type=int,   default=8)
    parser.add_argument("--data_root",  default=LOCAL_L2ARCTIC_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Naive LoRA Fine-Tuning  device={device} ===")

    print("Loading data ...")
    train_utts, dev_utts = load_train_dev_utterances(local_root=args.data_root)
    print(f"  train={len(train_utts)}  dev={len(dev_utts)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

    model = build_lora_model(MODEL_ID, r=args.lora_r)
    model.config.use_cache                        = False
    model.config.forced_decoder_ids               = None
    model.generation_config.suppress_tokens       = None
    model.generation_config.begin_suppress_tokens = None

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        refs  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=preds, references=refs)}

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = args.lr,
        num_train_epochs            = args.epochs,
        fp16                        = torch.cuda.is_available(),
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 25,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = 225,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
        report_to                   = "none",
        dataloader_num_workers      = 2,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = L2ArcticDataset(train_utts, processor),
        eval_dataset    = L2ArcticDataset(dev_utts,   processor),
        data_collator   = WhisperDataCollator(processor=processor),
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training ...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved → {args.output_dir}")

    # Plot loss curve
    plot_loss_curve(trainer.state.log_history, Path(args.output_dir) / "loss_curve.png")

    print("Done.")

if __name__ == "__main__":
    main()