# -*- coding: utf-8 -*-
"""
preprocess_audio.py

Preprocesses LibriSpeech audio files by:
1. Loading .wav files and resampling to 16kHz mono
2. Extracting hidden states using Whisper encoder
3. Saving hidden states (bfloat16) and transcripts as .pt files

Usage:
python -m src.data.preprocess_audio --source_dir <path> --output_dir <path> --model_name <model>
"""

import os
from pathlib import Path
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm
import argparse
from typing import Dict

def ensure_mono_and_resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Converts audio to mono and resamples to target sample rate."""
    # Convert to mono if needed
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform.squeeze(0)

def load_transcripts(root: Path) -> Dict[str, str]:
    """Loads all transcripts from .trans.txt files."""
    transcripts = {}
    for txt_file in root.rglob("*.trans.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
    return transcripts

def main(args: argparse.Namespace):
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = args.model_name
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name}'. Please check the model name and your internet connection.")
        print(e)
        return

    # Prepare paths
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found at {source_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.absolute()}")

    # Load transcripts
    print("Loading transcripts...")
    transcripts = load_transcripts(source_dir)
    print(f"Loaded {len(transcripts)} transcripts.")

    # Process audio files
    audio_files = list(source_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")

    if len(audio_files) != len(transcripts):
        audio_ids = {p.stem for p in source_dir.rglob("*.wav")}
        transcript_ids = set(transcripts.keys())

        missing_transcripts = audio_ids - transcript_ids
        missing_audio = transcript_ids - audio_ids

        print("Missing transcripts for:", missing_transcripts)
        print("Transcripts with no audio:", missing_audio)

    for audio_path in tqdm(audio_files, desc=f"Encoding {source_dir.name}"):
        utt_id = audio_path.stem

        # Preserve directory structure
        relative_path = audio_path.relative_to(source_dir)
        output_filepath = output_dir / relative_path.with_suffix('.pt')
        
        if output_filepath.exists():
            continue
            
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            processed_waveform = ensure_mono_and_resample(waveform, sr)
            
            # Extract hidden states
            inputs = processor(processed_waveform, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                hidden_states = model.encoder(input_features=input_features).last_hidden_state
                hidden_states = hidden_states.to(torch.bfloat16)  # Reduce file size
                hidden_states = hidden_states.squeeze(0).cpu()
            
            transcript_text = transcripts.get(utt_id, "")
            if not transcript_text:
                print(f"\nWarning: Transcript not found for {utt_id}")

            save_data = {
                "hidden_states": hidden_states, 
                "transcript": transcript_text
            }
            
            torch.save(save_data, output_filepath)

        except Exception as e:
            print(f"\nWarning: Could not process file {audio_path}. Error: {e}")
            continue

    print(f"\n✅ Pre-processing complete for partition '{source_dir.name}'.")
    print(f"All outputs are saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess LibriSpeech audio files using a Whisper encoder."
    )
    parser.add_argument(
        '--source_dir', 
        type=str, 
        required=True, 
        help='Path to the source LibriSpeech partition folder'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Path to save processed .pt files'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="openai/whisper-small", 
        help='Whisper model name from Hugging Face Hub'
    )
    
    args = parser.parse_args()
    main(args)