from pathlib import Path

DATASET_NAME = "KoelLabs/L2Arctic"
LOCAL_L2ARCTIC_DIR = Path("data/l2_arctic")

# Hold out one speaker per L1 for testing
# Even amount of f/m
HELD_OUT_SPEAKERS = {"SKA", "BWC", "SVBI", "HJK", "EBVS", "HQTV"}