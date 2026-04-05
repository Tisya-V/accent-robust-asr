from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import pandas as pd
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

from src.utils.audio_utils import bytes_to_array
from src.utils.load_l2arctic import load_scripted, split_dataset

MODEL_ID = "openai/whisper-small"


# ── Dataset ────────────────────────────────────────────────────────────────────

class L2ArcticDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: WhisperProcessor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio_array = bytes_to_array(row["audio"]["bytes"])
        inputs = self.processor(
            audio_array, sampling_rate=16000,
            return_tensors="pt", truncation=True,
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer(row["text"], return_tensors="pt").input_ids[0]
        return {
            "input_features": inputs.input_features[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels,
        }


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])
        attention_masks = torch.stack([f["attention_mask"] for f in features])
        label_list = [f["labels"] for f in features]
        max_len = max(l.shape[0] for l in label_list)
        padded_labels = torch.full((len(label_list), max_len), self.pad_token_id, dtype=torch.long)
        for i, lab in enumerate(label_list):
            padded_labels[i, :lab.shape[0]] = lab
        return {
            "input_features": input_features,
            "attention_mask": attention_masks,
            "labels": padded_labels,
        }


# ── Model ──────────────────────────────────────────────────────────────────────

def get_whisper_lora(model_id: str, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models/baseline_loraft")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_scripted()
    train_df, dev_df, _ = split_dataset(df)
    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

    model = get_whisper_lora(MODEL_ID, r=args.lora_r)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=preds, references=refs)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=L2ArcticDataset(train_df, processor),
        eval_dataset=L2ArcticDataset(dev_df, processor),
        data_collator=WhisperDataCollator(processor=processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()