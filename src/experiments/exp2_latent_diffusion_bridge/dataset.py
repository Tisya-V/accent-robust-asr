"""
BridgeDataset: load paired encoder states for diffusion bridge training.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.bridge_utils import get_split_data_dir


class BridgeDataset(Dataset):
    """
    Load (z_acc, z_nat) pairs from encoder state files.

    Encoder states are stored as bfloat16 tensors. We upcast to float32 for
    training stability (full precision during loss computation).

    Args:
        mapping_path: path to mapping JSON
        split:        "train" or "dev"
        alignment:    "position" (default) or "dtw" — if "dtw", loads precomputed
                      DTW paths from the dtw_path field in the mapping JSON
    """

    def __init__(self, mapping_path: str, split: str = "train", alignment: str = "position"):
        self.mapping_path = Path(mapping_path)
        self.split        = split
        self.alignment    = alignment

        with open(self.mapping_path) as f:
            self.pairs = json.load(f)

        if split not in ("train", "dev"):
            raise ValueError(f"Unknown split: {split}")

        print(f"[BridgeDataset] Loaded {len(self.pairs)} pairs from {self.mapping_path} "
              f"(alignment={alignment})")

        print("[BridgeDataset] Resolving encoder state paths...")
        self._resolved = self._resolve_all_paths()
        missing = sum(1 for l, n in self._resolved if l is None or n is None)
        print(f"[BridgeDataset] {len(self.pairs)} pairs ({missing} missing encoder states)")

        if alignment == "dtw":
            missing_dtw = sum(1 for p in self.pairs if not p.get("dtw_path"))
            if missing_dtw:
                raise RuntimeError(
                    f"{missing_dtw} pairs missing dtw_path — run precompute_dtw.py first."
                )

    def _find_file(self, rel_path: str) -> Optional[Path]:
        for split_name in ["train", "dev"]:
            p = get_split_data_dir(split_name) / rel_path
            if p.exists():
                return p
        return None

    def _resolve_all_paths(self):
        """Pre-resolve all encoder state paths using threads — NFS stat calls release the GIL."""
        unique = list({p for pair in self.pairs
                       for p in (pair["l2_encoder_state_path"], pair["nat_encoder_state_path"])})
        with ThreadPoolExecutor(max_workers=16) as ex:
            resolved = dict(zip(unique, ex.map(self._find_file, unique)))
        return [(resolved[pair["l2_encoder_state_path"]],
                 resolved[pair["nat_encoder_state_path"]])
                for pair in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        pair               = self.pairs[idx]
        l2_path, nat_path  = self._resolved[idx]

        if l2_path is None:
            raise FileNotFoundError(f"Missing L2 encoder state: {pair['l2_encoder_state_path']}")
        if nat_path is None:
            raise FileNotFoundError(f"Missing native encoder state: {pair['nat_encoder_state_path']}")

        l2_state  = torch.load(l2_path,  map_location="cpu", weights_only=False)
        nat_state = torch.load(nat_path, map_location="cpu", weights_only=False)

        z_acc = l2_state["hidden_states"]   # [1500, 768] bf16
        z_nat = nat_state["hidden_states"]  # [1500, 768] bf16

        assert z_acc.shape == (1500, 768), f"Unexpected z_acc shape: {z_acc.shape}"
        assert z_nat.shape == (1500, 768), f"Unexpected z_nat shape: {z_nat.shape}"

        l2_speech_end  = pair["l2_speech_end_frame"]
        nat_speech_end = pair["nat_speech_end_frame"]

        if self.alignment == "dtw":
            path_arr = np.load(pair["dtw_path"])  # [P, 2] int16
            return z_acc, z_nat, l2_speech_end, nat_speech_end, path_arr

        return z_acc, z_nat, l2_speech_end, nat_speech_end
