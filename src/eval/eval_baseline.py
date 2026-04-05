from __future__ import annotations

import argparse
import torch
from scipy.signal import resample_poly
from math import gcd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.audio_utils import bytes_to_array
from src.utils.load_l2arctic import load_scripted, load_spontaneous, split_dataset
from src.eval.eval_utils import (
    add_normalized_columns,
    compute_metrics_df,
    compute_grouped_metrics,
    attach_utterance_stats,
)

MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_dataset(df, processor, model, device, batch_size=8):
    predictions = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        audio_arrays = [bytes_to_array(row["audio"]["bytes"]) for _, row in batch_df.iterrows()]
        inputs = processor(
            audio_arrays, sampling_rate=16000,
            return_tensors="pt", truncation=True,
            return_attention_mask=True,
        )
        with torch.no_grad():
            pred_ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                language="en",
                task="transcribe",
            )
        predictions.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Whisper-small baseline eval")
    parser.add_argument("--spontaneous", action="store_true", help="Evaluate on spontaneous split (default: scripted)")
    args = parser.parse_args()

    split_name = "spontaneous" if args.spontaneous else "scripted"
    loader = load_spontaneous if args.spontaneous else load_scripted

    print(f"Loading {split_name} data...")
    df = loader()
    _, _, test_df = split_dataset(df)
    test_df = test_df.reset_index(drop=True)
    print(f"Test set: {len(test_df)} utterances")

    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None
    print(f"Running on: {DEVICE}")

    print("Transcribing...")
    predictions = transcribe_dataset(test_df, processor, model, device=DEVICE)

    results_df = test_df.drop(columns=["audio"]).copy()
    results_df["prediction"] = predictions
    results_df = add_normalized_columns(results_df, ref_col="text", pred_col="prediction")
    results_df = attach_utterance_stats(results_df)

    overall = compute_metrics_df(results_df)
    by_l1 = compute_grouped_metrics(results_df, group_col="speaker_native_language")

    print(f"\n=== Zero-shot Whisper-small [{split_name}] ===")
    print(f"  WER: {overall['wer']:.4f}")
    print(f"  CER: {overall['cer']:.4f}")
    print("\nBy L1:")
    print(by_l1.to_string(index=False))

    results_df.to_csv(f"results/baseline_zeroshot_{split_name}_predictions.csv", index=False)
    by_l1.to_csv(f"results/baseline_zeroshot_{split_name}_by_l1.csv", index=False)
    print(f"\nDone! Results saved to results/baseline_zeroshot_{split_name}_*.csv")


if __name__ == "__main__":
    main()