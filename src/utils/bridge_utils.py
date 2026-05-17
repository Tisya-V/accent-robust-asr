"""Utilities for E2 Latent Diffusion Bridge data loading."""

import os
from pathlib import Path


def get_split_data_dir(split: str) -> Path:
    """
    Get data directory for a split (train/dev/test).
    Respects environment variables TRAIN_DATA_DIR, DEV_DATA_DIR, TEST_DATA_DIR.

    Args:
        split: "train", "dev", or "test"

    Returns:
        Path to the split's data directory
    """
    env_map = {
        "train": ("TRAIN_DATA_DIR", "data/processed/train"),
        "dev": ("DEV_DATA_DIR", "data/processed/dev"),
        "test": ("TEST_DATA_DIR", "data/processed/test"),
    }

    if split not in env_map:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'dev', or 'test'.")

    env_var, default_path = env_map[split]
    path_str = os.environ.get(env_var, default_path)
    return Path(path_str)
