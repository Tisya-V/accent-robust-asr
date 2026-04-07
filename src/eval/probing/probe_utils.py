"""
probe_utils.py
Shared utilities for the L2-ARCTIC Whisper probing suite.

Responsibilities:
- Whisper encoder hidden-state extraction (all layers)
- TextGrid phoneme segment parsing
- Segment mean-pooling of encoder frames
- ARPAbet vocab and phonological feature mapping
- Shared data loading helpers
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import librosa
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_FRAME_RATE = 50          # Whisper: 50 Hz (20ms per frame)
WHISPER_HIDDEN_DIM = 512         # whisper-small
WHISPER_N_ENCODER_LAYERS = 6     # whisper-small has 6 encoder transformer layers

# 39 standard ARPAbet phones (excluding silence markers)
ARPABET_VOCAB = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH",
]
PHONE2ID = {p: i for i, p in enumerate(ARPABET_VOCAB)}
NUM_PHONES = len(ARPABET_VOCAB)

# Silence / pause tokens to skip
SILENCE_LABELS = {"SIL", "SP", "sil", "sp", "spn", "<eps>", ""}

# L2-ARCTIC speaker → L1 mapping (all 24 speakers)
SPEAKER_L1 = {
    "ABA": "Arabic",   "SKA": "Arabic",
    "YBAA": "Arabic",  "ZHAA": "Arabic",
    "LXC": "Chinese", "NCC": "Chinese",
    "BWC": "Chinese", "TXHC": "Chinese",
    "ASI": "Hindi",    "RRBI": "Hindi",
    "SVBI": "Hindi",   "TNI": "Hindi",
    "HJK": "Korean",   "HKK": "Korean",
    "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish",
    "MBMPS": "Spanish",  "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese",
    "THV": "Vietnamese", "TLV": "Vietnamese",
}

L1_GROUPS = sorted(set(SPEAKER_L1.values()))
L1_2_ID   = {l1: i for i, l1 in enumerate(L1_GROUPS)}
NUM_L1S   = len(L1_GROUPS)

# Phonological feature matrix: (NUM_PHONES, NUM_FEATURES)
# Features: [voiced, nasal, fricative, affricate, stop, approximant,
#             lateral, labial, coronal, dorsal, vowel, front, back, high, low, round]
_PHON_FEATURES = {
    # Vowels
    "AA": [1,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0],
    "AE": [1,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0],
    "AH": [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
    "AO": [1,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1],
    "AW": [1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0],
    "AY": [1,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0],
    "EH": [1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
    "ER": [1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
    "EY": [1,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0],
    "IH": [1,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0],
    "IY": [1,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0],
    "OW": [1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1],
    "OY": [1,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1],
    "UH": [1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1],
    "UW": [1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1],
    # Stops
    "B":  [1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    "D":  [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
    "G":  [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
    "K":  [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
    "P":  [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    "T":  [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
    # Fricatives
    "DH": [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "F":  [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
    "HH": [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
    "S":  [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "SH": [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "TH": [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "V":  [1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
    "Z":  [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "ZH": [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    # Affricates
    "CH": [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    "JH": [1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    # Nasals
    "M":  [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    "N":  [1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    "NG": [1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    # Approximants / liquids
    "L":  [1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
    "R":  [1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
    "W":  [1,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1],
    "Y":  [1,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0],
}
PHON_FEATURE_NAMES = [
    "voiced","nasal","fricative","affricate","stop","approximant",
    "lateral","labial","coronal","dorsal","vowel","front","back","high","low","round"
]
NUM_PHON_FEATURES = len(PHON_FEATURE_NAMES)

PHON_FEATURE_MATRIX = np.zeros((NUM_PHONES, NUM_PHON_FEATURES), dtype=np.float32)
for phone, feats in _PHON_FEATURES.items():
    if phone in PHONE2ID:
        PHON_FEATURE_MATRIX[PHONE2ID[phone]] = feats


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

@dataclass
class PhoneSegment:
    label: str          # ARPAbet label
    start: float        # seconds
    end:   float        # seconds
    phone_id: int = -1  # index into ARPABET_VOCAB (-1 if silence/unknown)

    @property
    def start_frame(self) -> int:
        return int(self.start * ENCODER_FRAME_RATE)

    @property
    def end_frame(self) -> int:
        return max(self.start_frame + 1, int(self.end * ENCODER_FRAME_RATE))

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[PhoneSegment]:
    """
    Parse a Praat TextGrid file and return a list of PhoneSegments.
    Skips silence tokens.  Handles both short-format and long-format TextGrids.
    """
    segments = []
    path = Path(textgrid_path)
    if not path.exists():
        raise FileNotFoundError(f"TextGrid not found: {textgrid_path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]

    # Find the target tier
    in_target_tier = False
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect tier header
        if 'name = "' in line or "name = " in line:
            tier_label = line.split("=", 1)[1].strip().strip('"')
            in_target_tier = (tier_label == tier_name)
        if in_target_tier and ("intervals [" in line or "intervals:" in line):
            # Try to parse a block: xmin, xmax, text
            try:
                xmin = float(lines[i+1].split("=")[1].strip())
                xmax = float(lines[i+2].split("=")[1].strip())
                text_val = lines[i+3].split("=", 1)[1].strip().strip('"')
                # Normalise ARPAbet label: strip stress digits
                label = text_val.rstrip("012").upper()
                if label not in SILENCE_LABELS:
                    pid = PHONE2ID.get(label, -1)
                    segments.append(PhoneSegment(
                        label=label, start=xmin, end=xmax, phone_id=pid
                    ))
                i += 4
                continue
            except (IndexError, ValueError):
                pass
        i += 1

    return segments


# ---------------------------------------------------------------------------
# Whisper hidden-state extraction
# ---------------------------------------------------------------------------

def extract_encoder_hidden_states(
    model,
    processor,
    audio_array: np.ndarray,
    sampling_rate: int = 16000,
    device: str = "cpu",
) -> Tuple[torch.Tensor, ...]:
    """
    Run Whisper encoder on a single audio array.
    Returns a tuple of tensors of shape (1, T, 512):
        index 0  → embedding output (after positional encoding)
        index 1  → layer 1 output
        ...
        index 6  → layer 6 output   (top encoder layer for whisper-small)

    So hidden_states[layer_idx + 1] gives the output of encoder layer `layer_idx`.
    """
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    model.eval()
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
    # tuple: (embed_output, layer_0_out, ..., layer_5_out) for whisper-small
    return encoder_outputs.hidden_states   # length = N_layers + 1


def pool_segment(
    hidden_states: torch.Tensor,  # (1, T, D)
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    """Mean-pool encoder frames over [start_frame, end_frame) → (D,) numpy array."""
    end_frame = min(end_frame, hidden_states.shape[1])
    if end_frame <= start_frame:
        return None
    seg = hidden_states[0, start_frame:end_frame, :].mean(dim=0)
    return seg.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Building the embedding dataset
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """One segment's embedding at a given layer, with all metadata."""
    embedding:  np.ndarray        # (D,)
    phone_id:   int
    phone_label: str
    speaker_id: str
    l1_id:      int
    l1_label:   str
    layer_idx:  int
    utterance_id: str


def build_embedding_dataset(
    model,
    processor,
    utterances: List[Dict],        # list of {audio, textgrid, speaker, utterance_id}
    layer_indices: Optional[List[int]] = None,
    device: str = "cpu",
    min_duration_ms: float = 30.0,  # skip segments shorter than this
) -> List[EmbeddingRecord]:
    """
    Extract segment-level embeddings for all utterances at specified encoder layers.

    Args:
        utterances: list of dicts with keys:
            - 'audio':        np.ndarray, 16kHz mono
            - 'textgrid':     path to TextGrid file
            - 'speaker':      speaker ID string (e.g. 'ABA')
            - 'utterance_id': unique string
        layer_indices: which encoder layers to extract (0 = embedding, 1-6 = transformer layers).
                       Default: all 7 (0..6).
        min_duration_ms: skip phone segments shorter than this.

    Returns:
        List of EmbeddingRecord objects (one per segment per layer).
    """
    if layer_indices is None:
        layer_indices = list(range(WHISPER_N_ENCODER_LAYERS + 1))  # 0..6

    records = []
    for utt in tqdm(utterances, desc="Extracting"):

        audio, sr = sf.read(utt["wav_path"], dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        speaker = utt["speaker"]
        l1 = SPEAKER_L1.get(speaker, "Unknown")
        l1_id = L1_2_ID.get(l1, -1)

        # Parse phone segments from TextGrid
        try:
            segments = parse_textgrid(utt["textgrid"])
        except Exception as e:
            print(f"  [WARN] Could not parse TextGrid {utt['textgrid']}: {e}")
            continue

        # Only keep segments with known ARPAbet labels
        valid_segs = [s for s in segments
                      if s.phone_id >= 0
                      and s.duration * 1000 >= min_duration_ms]
        if not valid_segs:
            continue

        # Extract hidden states for this utterance
        try:
            hidden_states_tuple = extract_encoder_hidden_states(
                model, processor, audio, device=device
            )
        except Exception as e:
            print(f"  [WARN] Encoder failed for {utt['utterance_id']}: {e}")
            continue

        # Pool each segment at each requested layer
        for layer_idx in layer_indices:
            hs = hidden_states_tuple[layer_idx]   # (1, T, D)
            for seg in valid_segs:
                emb = pool_segment(hs, seg.start_frame, seg.end_frame)
                if emb is None:
                    continue
                records.append(EmbeddingRecord(
                    embedding=emb,
                    phone_id=seg.phone_id,
                    phone_label=seg.label,
                    speaker_id=speaker,
                    l1_id=l1_id,
                    l1_label=l1,
                    layer_idx=layer_idx,
                    utterance_id=utt["utterance_id"],
                ))

    return records


def records_to_arrays(
    records: List[EmbeddingRecord],
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter records to one layer and return aligned numpy arrays.
    Returns: X (N, D), phone_ids (N,), l1_ids (N,), speaker_strs (N,)
    """
    subset = [r for r in records if r.layer_idx == layer_idx]
    X          = np.stack([r.embedding    for r in subset])
    phone_ids  = np.array([r.phone_id     for r in subset])
    l1_ids     = np.array([r.l1_id        for r in subset])
    speakers   = np.array([r.speaker_id   for r in subset])
    return X, phone_ids, l1_ids, speakers


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str):
    """Save a results dict to JSON, converting numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(f"Unserializable: {type(obj)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"  Saved results → {path}")

