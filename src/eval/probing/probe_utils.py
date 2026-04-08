"""
src/utils/probe_utils.py
Shared utilities for the L2-ARCTIC Whisper probing suite.

Responsibilities
----------------
- Whisper encoder hidden-state extraction (all layers)
- TextGrid phoneme segment parsing
- Segment mean-pooling of encoder frames
- Shared dataset-building helpers

Phoneme vocab and phonological features live in src/utils/phonology.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.config import (
    ENCODER_FRAME_RATE,
    L1_2_ID,
    SILENCE_LABELS,
    SPEAKER_L1,
    WHISPER_N_ENCODER_LAYERS,
)
from src.utils.phonology import (
    ARPABET_VOCAB,
    NUM_PHONES,
    NUM_PHON_FEATURES,
    PHON_FEATURE_MATRIX,
    PHON_FEATURE_NAMES,
    PHONE2ID,
    phone_to_features,
)
from src.utils.textgrid import parse_textgrid


# ---------------------------------------------------------------------------
# Whisper hidden-state extraction
# ---------------------------------------------------------------------------

def extract_encoder_hidden_states(
    model,
    processor,
    wav_path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, ...]:
    """
    Run the Whisper encoder on a single audio file.

    Returns a tuple of (WHISPER_N_ENCODER_LAYERS + 1) tensors, each (1, T, D):
        index 0      → embedding output (after conv + positional encoding)
        index 1..12  → transformer encoder layer outputs 1–12

    i.e. hidden_states[layer_idx] gives the output *after* layer (layer_idx - 1),
    and hidden_states[0] is the pre-transformer embedding.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    model.eval()
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
    # length = WHISPER_N_ENCODER_LAYERS + 1  (13 for whisper-small)
    return encoder_outputs.hidden_states


def pool_segment(
    hidden_states: torch.Tensor,  # (1, T, D)
    start_frame: int,
    end_frame: int,
) -> Optional[np.ndarray]:
    """Mean-pool encoder frames over [start_frame, end_frame) → (D,) float32."""
    end_frame = min(end_frame, hidden_states.shape[1])
    if end_frame <= start_frame:
        return None
    return hidden_states[0, start_frame:end_frame].mean(dim=0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Embedding dataset construction
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """One phone segment's embedding at a given encoder layer, with metadata."""
    embedding:    np.ndarray  # (D,)
    phone_id:     int
    phone_label:  str
    speaker_id:   str
    l1_id:        int
    l1_label:     str
    layer_idx:    int         # 0 = embed, 1-12 = transformer layers
    utterance_id: str
    split:        str         # "train" | "dev" | "test" | "ood"


def build_embedding_dataset(
    model,
    processor,
    utterances: List[Dict],
    layer_indices: Optional[List[int]] = None,
    device: str = "cpu",
    min_duration_ms: float = 30.0,
) -> List[EmbeddingRecord]:
    """
    Extract segment-level embeddings for all utterances at specified encoder layers.

    Parameters
    ----------
    model, processor : loaded Whisper model + processor
    utterances       : list of dicts from load_l2arctic (must have wav_path,
                       textgrid, speaker, utterance_id, split)
    layer_indices    : encoder layers to extract; default = all 13 (0..12)
    device           : "cpu" or "cuda"
    min_duration_ms  : skip segments shorter than this (ms)

    Returns
    -------
    List of EmbeddingRecord (one per valid segment per requested layer).
    """
    if layer_indices is None:
        layer_indices = list(range(WHISPER_N_ENCODER_LAYERS + 1))  # 0..12

    records: List[EmbeddingRecord] = []

    for utt in tqdm(utterances, desc="Extracting embeddings"):
        speaker = utt["speaker"]
        l1      = SPEAKER_L1.get(speaker, "Unknown")
        l1_id   = L1_2_ID.get(l1, -1)
        split   = utt.get("split", "unknown")

        try:
            segments = parse_textgrid(utt["textgrid"])
        except Exception as e:
            print(f"  [WARN] TextGrid parse failed {utt['textgrid']}: {e}")
            continue

        valid_segs = [
            s for s in segments
            if s.phone_id >= 0 and s.duration * 1000 >= min_duration_ms
        ]
        if not valid_segs:
            continue

        try:
            hs_tuple = extract_encoder_hidden_states(
                model, processor, utt["wav_path"], device=device
            )
        except Exception as e:
            print(f"  [WARN] Encoder failed {utt['utterance_id']}: {e}")
            continue

        for layer_idx in layer_indices:
            hs = hs_tuple[layer_idx]           # (1, T, D)
            for seg in valid_segs:
                emb = pool_segment(hs, seg.start_frame, seg.end_frame)
                if emb is None:
                    continue
                records.append(EmbeddingRecord(
                    embedding    = emb,
                    phone_id     = seg.phone_id,
                    phone_label  = seg.label,
                    speaker_id   = speaker,
                    l1_id        = l1_id,
                    l1_label     = l1,
                    layer_idx    = layer_idx,
                    utterance_id = utt["utterance_id"],
                    split        = split,
                ))

    return records


def records_to_arrays(
    records: List[EmbeddingRecord],
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter records to one layer and return aligned numpy arrays.

    Returns
    -------
    X           (N, D)  embeddings
    phone_ids   (N,)    int
    l1_ids      (N,)    int
    speakers    (N,)    str
    splits      (N,)    str
    """
    subset   = [r for r in records if r.layer_idx == layer_idx]
    X        = np.stack([r.embedding   for r in subset])
    phone_ids = np.array([r.phone_id    for r in subset])
    l1_ids   = np.array([r.l1_id       for r in subset])
    speakers = np.array([r.speaker_id  for r in subset])
    splits   = np.array([r.split       for r in subset])
    return X, phone_ids, l1_ids, speakers, splits


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Save a results dict to JSON, converting numpy scalars/arrays."""
    def _convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        raise TypeError(f"Unserializable type: {type(obj)}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Saved → {path}")
