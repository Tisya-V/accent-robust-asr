"""
LatentCorrectorTransformer: plain Transformer encoder for z_acc -> z_nat mapping.
No diffusion, no timestep conditioning. Position-based alignment (frame i -> frame i).

Uses manual F.scaled_dot_product_attention (flash attention) matching the bridge architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with direct flash attention (no timestep conditioning)."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        normed = self.norm1(x)
        q = self.q_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class LatentCorrectorTransformer(nn.Module):
    """
    Maps accented encoder latents z_acc -> predicted native latents z_nat.

    Uses residual prediction: output = z_acc + delta, so the model only
    learns what to change, not to reconstruct from scratch.

    Input:  z_acc [B, 1500, 768]
    Output: z_nat_pred [B, 1500, 768]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        latent_dim: int = 768,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.proj_in = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, latent_dim)

        # Zero-init: model starts as identity, loss starts at MSE(z_acc, z_nat)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_acc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_acc: [B, 1500, 768] accented encoder latents
        Returns:
            z_nat_pred: [B, 1500, 768]
        """
        x = self.proj_in(z_acc)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return z_acc + self.proj_out(x)
