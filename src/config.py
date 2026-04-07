from pathlib import Path

DATASET_NAME = "KoelLabs/L2Arctic"
LOCAL_L2ARCTIC_DIR = Path("data/l2_arctic")

# Hold out one speaker per L1 for testing
# Even amount of f/m
HELD_OUT_SPEAKERS = {"SKA", "BWC", "SVBI", "HJK", "EBVS", "HQTV"}


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
SPEAKERS = sorted(SPEAKER_L1.keys())
L1_GROUPS = sorted(set(SPEAKER_L1.values()))
L1_2_ID   = {l1: i for i, l1 in enumerate(L1_GROUPS)}
NUM_L1S   = len(L1_GROUPS)

ENCODER_FRAME_RATE = 50          # Whisper: 50 Hz (20ms per frame)
WHISPER_HIDDEN_DIM = 512         # whisper-small
WHISPER_N_ENCODER_LAYERS = 6     # whisper-small has 6 encoder transformer layers

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