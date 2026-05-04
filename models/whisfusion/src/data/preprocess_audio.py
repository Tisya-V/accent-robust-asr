# -*- coding: utf-8 -*-
"""
preprocess_audio.py

Preprocesses LibriSpeech-style audio files by:
1. Loading .wav files and resampling to 16kHz mono
2. Extracting hidden states using Whisper encoder
3. Saving hidden states and transcripts as .pt files

Usage:
python -m src.data.preprocess_audio --source_dir <path> --output_dir <path> --model_name <model>
"""

from pathlib import Path
import argparse
from typing import Dict, Tuple

import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel


TARGET_SR = 16000


def ensure_mono_and_resample(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = TARGET_SR,
    resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] | None = None,
) -> torch.Tensor:
    """Convert audio to mono and resample to target sample rate."""
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != target_sr:
        key = (orig_sr, target_sr)
        if resamplers is not None:
            if key not in resamplers:
                resamplers[key] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=target_sr,
                )
            waveform = resamplers[key](waveform)
        else:
            waveform = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr,
            )(waveform)

    return waveform.squeeze(0).contiguous()


def load_transcripts(root: Path) -> Dict[str, str]:
    """Load all transcripts from .trans.txt files."""
    transcripts: Dict[str, str] = {}
    for txt_file in root.rglob("*.trans.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
    return transcripts


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found at {source_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.absolute()}")

    print("Loading transcripts...")
    transcripts = load_transcripts(source_dir)
    print(f"Loaded {len(transcripts)} transcripts.")

    print("Loading Whisper model...")
    try:
        processor = WhisperProcessor.from_pretrained(args.model_name)
        model = WhisperModel.from_pretrained(args.model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{args.model_name}'. Please check the model name and your internet connection.")
        print(e)
        return

    save_dtype = torch.bfloat16 if args.save_dtype == "bfloat16" else torch.float16

    wav_files = list(source_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} audio files to process.")

    resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

    with torch.inference_mode():
        for wav_path in tqdm(wav_files, desc=f"Encoding {source_dir.name}"):
            utt_id = wav_path.stem

            relative_path = wav_path.relative_to(source_dir)
            output_filepath = output_dir / relative_path.with_suffix(".pt")

            if output_filepath.exists():
                continue

            output_filepath.parent.mkdir(parents=True, exist_ok=True)

            try:
                waveform, sr = torchaudio.load(wav_path)
                processed_waveform = ensure_mono_and_resample(
                    waveform,
                    sr,
                    target_sr=TARGET_SR,
                    resamplers=resamplers,
                )

                inputs = processor(
                    processed_waveform,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.to(device, non_blocking=True)

                hidden_states = model.encoder(input_features=input_features).last_hidden_state
                hidden_states = hidden_states.to(save_dtype).squeeze(0).cpu()

                transcript_text = transcripts.get(utt_id, "")
                if not transcript_text:
                    print(f"\nWarning: Transcript not found for {utt_id}")

                torch.save(
                    {
                        "hidden_states": hidden_states,
                        "transcript": transcript_text,
                    },
                    output_filepath,
                )

            except Exception as e:
                print(f"\nWarning: Could not process file {wav_path}. Error: {e}")
                continue

    print(f"\n✅ Pre-processing complete for partition '{source_dir.name}'.")
    print(f"All outputs are saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess LibriSpeech-style audio files using a Whisper encoder."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source partition folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save processed .pt files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Whisper model name from Hugging Face Hub",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="dtype used for saving hidden states",
    )

    args = parser.parse_args()
    main(args)