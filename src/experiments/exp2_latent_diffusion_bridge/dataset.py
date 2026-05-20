"""
BridgeDataset: load paired encoder states for diffusion bridge training.
"""

import json
from concurrent.futures import ThreadPoolExecutor
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
        self.mapping_path = Path(mapping_path)
        self.split = split

        with open(self.mapping_path) as f:
            self.pairs = json.load(f)

        if split == "train":
            self.data_dir = get_split_data_dir("train")
        elif split == "dev":
            self.data_dir = get_split_data_dir("dev")
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"[BridgeDataset] Loaded {len(self.pairs)} pairs from {self.mapping_path}")
        print(f"[BridgeDataset] Resolving file paths...")
        self._resolved = self._resolve_all_paths()
        missing = sum(1 for l, n in self._resolved if l is None or n is None)
        print(f"[BridgeDataset] Resolved {len(self.pairs)} pairs ({missing} missing)")

    def _find_file(self, rel_path: str) -> Optional[Path]:
        for split_name in ["train", "dev"]:
            p = get_split_data_dir(split_name) / rel_path
            if p.exists():
                return p
        return None

    def _resolve_all_paths(self):
        """Pre-resolve all file paths using threads — NFS stat calls release the GIL."""
        unique = list({p for pair in self.pairs
                       for p in (pair["l2_encoder_state_path"], pair["nat_encoder_state_path"])})
        with ThreadPoolExecutor(max_workers=16) as ex:
            resolved = dict(zip(unique, ex.map(self._find_file, unique)))
        return [(resolved[pair["l2_encoder_state_path"]],
                 resolved[pair["nat_encoder_state_path"]])
                for pair in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pair = self.pairs[idx]
        l2_path, nat_path = self._resolved[idx]

        if l2_path is None:
            raise FileNotFoundError(f"Missing L2 encoder state: {pair['l2_encoder_state_path']}")
        if nat_path is None:
            raise FileNotFoundError(f"Missing native encoder state: {pair['nat_encoder_state_path']}")

        # Load encoder states
        try:
            l2_state = torch.load(l2_path, map_location="cpu", weights_only=False)
            nat_state = torch.load(nat_path, map_location="cpu", weights_only=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing encoder state: {e}")

        # Extract hidden states (keep in bf16 for memory efficiency)
        z_acc = l2_state["hidden_states"]  # [1500, 768] in bf16
        z_nat = nat_state["hidden_states"]  # [1500, 768] in bf16

        # Sanity checks
        assert z_acc.shape == (1500, 768), f"Unexpected z_acc shape: {z_acc.shape}"
        assert z_nat.shape == (1500, 768), f"Unexpected z_nat shape: {z_nat.shape}"

        speech_end = pair["speech_end_frame"]

        return z_acc, z_nat, speech_end
