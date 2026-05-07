"""
MiniMDM: wrapper around TransEncoder configured for small size.
Reuses the existing diffusion model architecture from src/training/lit_gpt/diffmodel.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src/training to path for lit_gpt imports
_training_dir = Path(__file__).parent.parent.parent / "training"
if str(_training_dir) not in sys.path:
    sys.path.insert(0, str(_training_dir))

from lit_gpt.diffmodel import TransEncoder
from lit_gpt.config import Config


def create_mini_mdm(
    vocab_size: int = 32000,
    n_embd: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    encoder_dim: int = 768,
) -> TransEncoder:
    """
    Create a small TransEncoder configured for token correction.

    Args:
        vocab_size: TinyLlama vocab size (padded)
        n_embd: embedding dimension
        n_layers: number of transformer layers
        n_heads: number of attention heads
        encoder_dim: dimension of Whisper encoder states (unused, for clarity)

    Returns:
        TransEncoder instance ready for training
    """
    config = Config(
        n_layer=n_layers,
        n_embd=n_embd,
        n_head=n_heads,
        padded_vocab_size=vocab_size,
        block_size=256,
        bias=False,
        norm_class=nn.LayerNorm,
        norm_eps=1e-5,
        mlp_class=None,  # Will be set to default
        shared_attention_norm=False,
        rotary_percentage=1.0,
        condense_ratio=1,
        n_query_groups=n_heads,
        head_size=n_embd // n_heads,
    )

    return TransEncoder(config)
