"""
BridgeTransformer: bidirectional Transformer for latent diffusion.
Maps [B, 1500, 768] encoder states with timestep conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Convert scalar timestep t ∈ [0,1] to conditioned embeddings."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Sinusoidal position encoding for timestep
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps in [0, 1]

        Returns:
            emb: [B, d_model] timestep embeddings
        """
        # Expand to [B, 1]
        t = t.unsqueeze(-1).float()

        # Pass through MLP
        emb = self.time_embedding(t)

        return emb


class AdaLNBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning on timestep.

    Structure:
    1. Pre-norm self-attention (adaptive layer norm from timestep)
    2. Residual
    3. Pre-norm MLP (adaptive layer norm from timestep)
    4. Residual
    """

    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention (bidirectional, no causal masking)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

        # AdaLN-Zero: per-layer scale/shift from timestep embedding
        # Initialized to zero so model outputs zero (identity) at init
        self.scale1 = nn.Parameter(torch.zeros(d_model))
        self.scale2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] sequence
            t_emb: [B, D] timestep embedding

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape

        # Self-attention block
        # Pre-norm then attention
        normed = self.norm1(x)  # [B, L, D]

        # AdaLN-Zero: scale by timestep-dependent factor
        # Broadcast t_emb [B, D] to [B, L, D]
        scale = 1.0 + self.scale1 * t_emb.unsqueeze(1)
        normed = normed * scale

        # Self-attention
        attn_out, _ = self.attn(normed, normed, normed)

        # Residual
        x = x + attn_out

        # MLP block
        normed = self.norm2(x)  # [B, L, D]

        # AdaLN-Zero: scale by timestep-dependent factor
        scale = 1.0 + self.scale2 * t_emb.unsqueeze(1)
        normed = normed * scale

        # MLP
        mlp_out = self.mlp(normed)

        # Residual
        x = x + mlp_out

        return x


class BridgeTransformer(nn.Module):
    """
    Lightweight bidirectional Transformer for latent diffusion bridge.

    Input: [B, 1500, 768] (accented encoder latents) + timestep t
    Output: [B, 1500, 768] (predicted noise for diffusion)

    No projections: operates directly in 768-dim for simplicity.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)

        # Stack of AdaLN blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise for diffusion.

        Args:
            z_t: [B, 1500, 768] noisy encoder latents at timestep t
            t: [B] timesteps in [0, 1]

        Returns:
            eps_pred: [B, 1500, 768] predicted noise (same shape as input)
        """
        B, L, D = z_t.shape

        # Embed timestep
        t_emb = self.time_embedding(t)  # [B, D]

        # Pass through AdaLN blocks
        x = z_t
        for block in self.blocks:
            x = block(x, t_emb)

        # Final norm
        x = self.norm_final(x)

        # Output: predicted noise (same shape as input)
        # No output projection needed; operate in native 768-dim
        eps_pred = x

        return eps_pred
