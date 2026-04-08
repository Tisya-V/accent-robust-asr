"""
src/utils/phonology.py
ARPAbet vocabulary and articulatory feature vectors.

Dependencies: phonecodes (ARPAbet → IPA), panphon (IPA → feature vectors).
Nothing is manually defined here — all feature values come from panphon.

pip install phonecodes panphon

Public API
----------
ARPABET_VOCAB        list[str]              39 phones, canonical order
PHONE2ID             dict[str, int]
NUM_PHONES           int
PHON_FEATURE_NAMES   list[str]              panphon feature names
NUM_PHON_FEATURES    int
PHON_FEATURE_MATRIX  np.ndarray (39, F)     float32, from panphon
phone_to_features(p)              -> np.ndarray (F,)
phones_to_feature_matrix(phones)  -> np.ndarray (N, F)
"""

from __future__ import annotations

import panphon
from phonecodes import phonecodes
import numpy as np

# ---------------------------------------------------------------------------
# ARPAbet vocabulary — 39 phones, silence excluded
# ---------------------------------------------------------------------------

ARPABET_VOCAB: list[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH",
]
PHONE2ID:   dict[str, int] = {p: i for i, p in enumerate(ARPABET_VOCAB)}
NUM_PHONES: int             = len(ARPABET_VOCAB)


# ---------------------------------------------------------------------------
# Build feature matrix via phonecodes + panphon at import time
# ---------------------------------------------------------------------------

def _arpabet_to_ipa(phone: str) -> str:
    """Convert a single stress-free ARPAbet phone to an IPA string via phonecodes."""
    # phonecodes expects space-separated tokens
    return phonecodes.convert(phone, "arpabet", "ipa", "eng").strip()


def _build_feature_matrix() -> tuple[np.ndarray, list[str]]:
    ft         = panphon.FeatureTable()
    feat_names = list(ft.names)                 # e.g. 22 feature name strings

    matrix = np.zeros((NUM_PHONES, len(feat_names)), dtype=np.float32)

    for phone in ARPABET_VOCAB:
        idx = PHONE2ID[phone]
        ipa = _arpabet_to_ipa(phone)            # e.g. "AE" -> "æ", "AW" -> "aʊ"
        vecs = ft.word_to_vector_list(ipa, numeric=True)
        # word_to_vector_list returns one list per IPA segment; values in {-1, 0, +1}
        # mean-pool across segments to handle diphthongs / affricates uniformly
        if vecs:
            matrix[idx] = np.array(vecs, dtype=np.float32).mean(axis=0)

    return matrix, feat_names


PHON_FEATURE_MATRIX, PHON_FEATURE_NAMES = _build_feature_matrix()
NUM_PHON_FEATURES: int = len(PHON_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def phone_to_features(phone: str) -> np.ndarray:
    """(NUM_PHON_FEATURES,) float32 feature vector for one ARPAbet phone."""
    return PHON_FEATURE_MATRIX[PHONE2ID[phone]].copy()


def phones_to_feature_matrix(phones: list[str]) -> np.ndarray:
    """(len(phones), NUM_PHON_FEATURES) matrix. Unknown phones → zero vector."""
    out = np.zeros((len(phones), NUM_PHON_FEATURES), dtype=np.float32)
    for i, p in enumerate(phones):
        if p in PHONE2ID:
            out[i] = PHON_FEATURE_MATRIX[PHONE2ID[p]]
    return out