"""
BridgeDataset: load paired encoder states for diffusion bridge training.
"""

import json
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset

from src.utils.bridge_utils import get_split_data_dir


class BridgeDataset(Dataset):
    """
    Load (z_acc, z_nat) pairs from encoder state files.

    Encoder states are stored as bfloat16 tensors. We upcast to float32 for
    training stability (full precision during loss computation).
    """

    def __init__(self, mapping_path: str, split: str = "train"):
        """
        Args:
            mapping_path: path to mapping JSON (e.g., mapping_train.json)
            split: "train" or "dev" — indicates which split the mapping covers
                   (used to find the correct data directory)
        """
        self.mapping_path = Path(mapping_path)
        self.split = split

        # Load mapping
        with open(self.mapping_path) as f:
            self.pairs = json.load(f)

        # Get data directory for this split (from env var or default)
        if split == "train":
            self.data_dir = get_split_data_dir("train")
        elif split == "dev":
            self.data_dir = get_split_data_dir("dev")
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"[BridgeDataset] Loaded {len(self.pairs)} pairs from {self.mapping_path}")
        print(f"[BridgeDataset] Data dir: {self.data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Load a pair of encoder states.

        Returns:
            z_acc: [1500, 768] float32 accented encoder hidden state
            z_nat: [1500, 768] float32 native encoder hidden state
            speech_end: int frame index where speech ends (for masking)
        """
        pair = self.pairs[idx]

        # Construct full paths
        l2_path = self.data_dir / pair["l2_encoder_state_path"]
        nat_path = self.data_dir / pair["nat_encoder_state_path"]

        # Load encoder states
        try:
            l2_state = torch.load(l2_path, map_location="cpu")
            nat_state = torch.load(nat_path, map_location="cpu")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing encoder state: {e}")

        # Extract hidden states and upcast bf16 → float32
        z_acc = l2_state["hidden_states"].float()  # [1500, 768]
        z_nat = nat_state["hidden_states"].float()  # [1500, 768]

        # Sanity checks
        assert z_acc.shape == (1500, 768), f"Unexpected z_acc shape: {z_acc.shape}"
        assert z_nat.shape == (1500, 768), f"Unexpected z_nat shape: {z_nat.shape}"

        speech_end = pair["speech_end_frame"]

        return z_acc, z_nat, speech_end
