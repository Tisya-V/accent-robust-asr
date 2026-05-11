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
feature_edit_distance(mat_i, mat_j, indel_cost) -> float

"""

from __future__ import annotations

import panphon
from phonecodes import phonecodes
import numpy as np
import numba


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


@numba.njit(cache=True)
def feature_edit_distance(
    mat_i: np.ndarray,   # (Li, F) float32
    mat_j: np.ndarray,   # (Lj, F) float32
    indel_cost: float = 0.5,
) -> float:
    """
    Articulatory feature edit distance between two phoneme sequences.

    Each row of mat_i / mat_j is a feature vector for one phoneme
    (from phones_to_feature_matrix). Substitution cost between two
    phonemes is their mean absolute feature difference, normalised to
    [0, 1]. Insertion / deletion cost is fixed at indel_cost.

    Returns a non-negative float. Lower = more phonemically similar.
    """
    Li = mat_i.shape[0]
    Lj = mat_j.shape[0]
    F  = mat_i.shape[1]

    # Allocate DP table
    dp = np.zeros((Li + 1, Lj + 1), dtype=np.float32)

    # Base cases: aligning against empty sequence = all indels
    for a in range(Li + 1):
        dp[a, 0] = a * indel_cost
    for b in range(Lj + 1):
        dp[0, b] = b * indel_cost

    # Fill DP table
    for a in range(1, Li + 1):
        for b in range(1, Lj + 1):
            # Mean absolute feature difference, normalised to [0, 1]
            # Divide by F to get mean, divide by 2 because max |diff| = 2 (i.e. -1 vs +1)
            sub = 0.0
            for f in range(F):
                sub += abs(mat_i[a - 1, f] - mat_j[b - 1, f])
            sub = (sub / F) / 2.0

            dp[a, b] = min(
                dp[a - 1, b]     + indel_cost,   # deletion
                dp[a, b - 1]     + indel_cost,   # insertion
                dp[a - 1, b - 1] + sub,          # substitution
            )

    return dp[Li, Lj]