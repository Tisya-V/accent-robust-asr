from pathlib import Path

import nltk as _nltk
NLTK_DATA_PATH = "/vol/bitbucket/tsv22/accent-robust-asr/nltk_data"
_nltk.data.path.insert(0, NLTK_DATA_PATH)


# Random seed for reproducibility
RANDOM_SEED = 42

# Model
MODEL_ID           = "openai/whisper-small"
ENCODER_FRAME_RATE = 50           # Whisper: 50 Hz (20ms per frame)
WHISPER_HIDDEN_DIM = 768          # whisper-small
WHISPER_N_ENCODER_LAYERS = 12     # whisper-small 

# Dataset
LOCAL_L2ARCTIC_DIR = Path("data/l2_arctic")
SUITCASE_SUBDIR = "suitcase_corpus"


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

SPEAKERS = sorted(SPEAKER_L1.keys())
# Hold out one speaker per L1 for testing
# Even amount of f/m
TEST_SPEAKERS = {"SKA", "BWC", "SVBI", "HJK", "EBVS", "HQTV"}
TRAIN_SPEAKERS = set(SPEAKERS) - TEST_SPEAKERS
L1_GROUPS = sorted(set(SPEAKER_L1.values()))
L1_2_ID   = {l1: i for i, l1 in enumerate(L1_GROUPS)}
NUM_L1S   = len(L1_GROUPS)


# Silence / pause tokens to skip
SILENCE_LABELS = {"SIL", "SP", "sil", "sp", "spn", "<eps>", ""}

# Linguistically motivated selection — phones most affected by L1 transfer
PROBE_PHONES = {
    "T",   # alveolar stop — aspiration/VOT varies hugely across L1s
    "D",   # voiced stop — devoicing common in Arabic, Korean, Hindi
    "R",   # rhotic — major L1 transfer site (tapped/trilled vs approximant)
    "L",   # lateral — dark L variation, Korean L/R merger
    "AE",  # low front vowel — absent in many L1 phoneme inventories
    "TH",  # dental fricative — notoriously absent outside English
    "S",   # sibilant — place of articulation varies by L1
}

# ---------------------------------------------------------------------------
# Results directories
# ---------------------------------------------------------------------------

RESULTS_DIR         = Path("results")
PROBE_RESULTS_DIR   = RESULTS_DIR / "probe_analysis"
CLUSTER_RESULTS_DIR = RESULTS_DIR / "clustering"
MODEL_PERF_COMPARISON_DIR = RESULTS_DIR / "model_perf_comparison"