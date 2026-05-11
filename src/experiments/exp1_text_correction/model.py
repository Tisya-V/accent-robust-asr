"""
MiniMDM: wrapper around TransEncoder configured for small size.
Reuses the existing diffusion model architecture from src/training/lit_gpt/diffmodel.py
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
from pathlib import Path

# Add src/training to path for lit_gpt imports
_training_dir = Path(__file__).parent.parent.parent / "training"
if str(_training_dir) not in sys.path:
    sys.path.insert(0, str(_training_dir))

from lit_gpt.diffmodel import TransEncoder
from lit_gpt.config import Config


class TransEncoderWithConditionProj(nn.Module):
    """Wrapper that adds condition projection to TransEncoder."""
    def __init__(self, encoder: TransEncoder, encoder_dim: int, n_embd: int):
        super().__init__()
        self.encoder = encoder
        self.condition_proj = nn.Linear(encoder_dim, n_embd, bias=False)

    def forward(self, idx: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            condition = self.condition_proj(condition)
        return self.encoder(idx, condition=condition)

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        self.condition_proj = self.condition_proj.to(*args, **kwargs)
        return self


def create_mini_mdm(
    vocab_size: int = 32000,
    n_embd: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    encoder_dim: int = 768,
) -> nn.Module:
    """
    Create a small TransEncoder configured for token correction.

    Args:
        vocab_size: TinyLlama vocab size (padded)
        n_embd: embedding dimension
        n_layers: number of transformer layers
        n_heads: number of attention heads
        encoder_dim: dimension of Whisper encoder states

    Returns:
        Model with condition projection layer ready for training
    """
    config = Config(
        n_layer=n_layers,
        n_embd=n_embd,
        n_head=n_heads,
        padded_vocab_size=vocab_size,
        block_size=256,
        bias=False,
        norm_eps=1e-5,
        shared_attention_norm=False,
        rotary_percentage=1.0,
        condense_ratio=1,
        n_query_groups=n_heads,
    )

    encoder = TransEncoder(config)
    return TransEncoderWithConditionProj(encoder, encoder_dim, n_embd)
