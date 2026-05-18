"""
BridgeTransformer: bidirectional Transformer for latent diffusion.
Maps [B, 1500, 768] encoder states with timestep conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding + learned MLP for timesteps t ∈ [0,1]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learned MLP to transform sinusoidal features (as in DiT)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding + learned transformation.

        1. Compute sinusoidal features: PE(t, 2i) = sin(t * w_i), PE(t, 2i+1) = cos(t * w_i)
           where w_i = 1 / 10000^(2i/d_model)
        2. Pass through learned MLP to get rich timestep representation

        Args:
            t: [B] timesteps in [0, 1]

        Returns:
            emb: [B, d_model] learned timestep embeddings
        """
        device, dtype = t.device, t.dtype
        B = t.shape[0]

        # Compute frequencies: w_i = 1 / 10000^(2i/d_model)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2, device=device, dtype=dtype) / self.d_model))

        # Compute phases: t_scaled [B, d_model/2] = t [B, 1] * freqs [d_model/2]
        t_scaled = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, d_model/2]

        # Interleave sin and cos: [sin(w_0*t), cos(w_0*t), sin(w_1*t), cos(w_1*t), ...]
        emb = torch.zeros(B, self.d_model, device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(t_scaled)  # Even indices: sin
        emb[:, 1::2] = torch.cos(t_scaled)  # Odd indices: cos

        # Learned transformation (MLP) — gives each block a rich, trainable t representation
        return self.mlp(emb)


class AdaLNBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning on timestep.
    Uses flash attention (via F.scaled_dot_product_attention) for efficiency.

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
        self.head_dim = d_model // n_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention components (manual for flash attention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Feedforward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

        # AdaLN-Zero: timestep embedding → (gamma1, beta1, gamma2, beta2)
        # Projects t_emb [B, D] to [B, 4*D], then chunk into 4 [B, D] tensors
        self.adaLN_proj = nn.Linear(d_model, 4 * d_model)

        # Initialize to zero: at initialization, gamma ≈ 1 (scale), beta ≈ 0 (shift)
        # This makes the model start as identity (no timestep conditioning) and gradually learn it
        nn.init.constant_(self.adaLN_proj.weight, 0)
        nn.init.constant_(self.adaLN_proj.bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] sequence
            t_emb: [B, D] timestep embedding

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape

        # Predict AdaLN params from timestep embedding
        adaLN_params = self.adaLN_proj(t_emb)  # [B, 4*D]
        gamma1, beta1, gamma2, beta2 = adaLN_params.chunk(4, dim=-1)  # each [B, D]

        # Self-attention block with flash attention
        normed = self.norm1(x)  # [B, L, D]

        # AdaLN-Zero: modulate with (1 + gamma) * scale + beta * shift
        normed = normed * (1.0 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)  # [B, L, D]

        # Project to Q, K, V
        q = self.q_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        k = self.k_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention (memory-efficient, uses fused kernels)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = x + attn_out

        # MLP block
        normed = self.norm2(x)  # [B, L, D]

        # AdaLN-Zero: modulate with (1 + gamma) * scale + beta * shift
        normed = normed * (1.0 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)  # [B, L, D]

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

    Projects 768-dim input to d_model for processing, then back to 768-dim.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        latent_dim: int = 768,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.latent_dim = latent_dim

        # Input projection: latent_dim (768) -> d_model
        self.proj_in = nn.Linear(latent_dim, d_model) if d_model != latent_dim else nn.Identity()

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)

        # Stack of AdaLN blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)

        # Output projection: d_model -> latent_dim (768)
        self.proj_out = nn.Linear(d_model, latent_dim) if d_model != latent_dim else nn.Identity()

        # Zero-initialize proj_out: at init, model predicts zero noise
        # This ensures initial loss is exactly the variance of the target (clean baseline)
        if d_model != latent_dim:
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

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

        # Project input to d_model
        x = self.proj_in(z_t)  # [B, L, d_model]

        # Embed timestep
        t_emb = self.time_embedding(t)  # [B, d_model]

        # Pass through AdaLN blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final norm
        x = self.norm_final(x)

        # Project back to latent_dim
        eps_pred = self.proj_out(x)  # [B, L, 768]

        return eps_pred
