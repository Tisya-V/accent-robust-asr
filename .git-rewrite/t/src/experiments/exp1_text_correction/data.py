"""
Exp1Dataset: loads cached encoder states + transcripts from .pt files.
Creates masked input sequences (simulating what SMDM produces mid-decoding).
Perturbation happens in forward_process() during training.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import random

from transformers import AutoTokenizer
from src.experiments.exp1_text_correction.config import DEFAULT_TOKENIZER_NAME


class Exp1Dataset(Dataset):
    """
    Dataset for token correction pretraining (Stage 1).
    Loads .pt files (encoder states + transcript), creates masked input sequences.
    Returns: {condition, target_ids, masked_ids, mask_indices}
    where masked_ids is the input to MiniMDM and mask_indices tells us which positions are masked.
    """

    def __init__(
        self,
        pt_files: List[Path],
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        max_length: int = 256,
        mask_ratio_range: Tuple[float, float] = (0.7, 1.0),
    ):
        super().__init__()
        self.pt_files = pt_files
        self.max_length = max_length
        self.mask_ratio_range = mask_ratio_range

        print(f"[Exp1Dataset] Using {len(self.pt_files)} utterances")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.vocab_size

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pt_path = self.pt_files[idx]

        # Load .pt file: {hidden_states, transcript}
        data = torch.load(pt_path, map_location="cpu")
        condition = data["hidden_states"].float()  # (1500, 768)
        text = data["transcript"]  # str

        # Tokenize transcript
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Truncate/pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = F.pad(tokens, (0, self.max_length - len(tokens)), value=self.tokenizer.pad_token_id)

        # Create masked input: sample a random mask ratio, then mask
        min_mask, max_mask = self.mask_ratio_range
        mask_ratio = random.uniform(min_mask, max_mask)

        # Create mask indices: positions where rand < mask_ratio get masked
        mask_indices = torch.rand(self.max_length) < mask_ratio

        # Create masked input
        masked_ids = tokens.clone()
        masked_ids[mask_indices] = self.mask_token_id

        return {
            "condition": condition,  # (1500, 768)
            "target_ids": tokens,  # (max_length,) — clean tokens
            "masked_ids": masked_ids,  # (max_length,) — with masked positions set to mask token
            "mask_indices": mask_indices,  # (max_length,) — True = masked, False = visible
        }


def create_dataloaders(
    batch_size: int = 16,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    max_length: int = 256,
    data_root: str = "data/processed",
    mask_ratio_range: Tuple[float, float] = (0.7, 1.0),
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    Scans data_root for .pt files, splits by train/dev directories.
    """
    data_root = Path(data_root)

    # Collect .pt files from train/ and dev/
    train_files = sorted((data_root / "train").rglob("*.pt"))
    dev_files = sorted((data_root / "dev").rglob("*.pt"))

    print(f"[create_dataloaders] Found {len(train_files)} train, {len(dev_files)} dev utterances")

    train_dataset = Exp1Dataset(
        train_files,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        mask_ratio_range=mask_ratio_range,
    )

    val_dataset = Exp1Dataset(
        dev_files,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        mask_ratio_range=mask_ratio_range,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
