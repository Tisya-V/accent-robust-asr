"""
BridgeTransformer: bidirectional Transformer for latent diffusion.
Maps [B, 1500, 768] encoder states with timestep conditioning.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary_embedding import build_rope_cache, apply_rotary_emb_func


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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:      [B, L, D] sequence
            t_emb:  [B, D] timestep embedding
            cos, sin: [L, head_dim/2] RoPE tables, pre-sliced to this sequence's length
            key_padding_mask: optional [B, L] bool, True = attend to this key position,
                False = padding to exclude. Broadcasts over heads and query positions —
                proven equivalent to cropping the sequence to the real length (see SDPA
                call below); lets a batch mix sequences of different lengths exactly.

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

        # Project to Q, K, V. Keep (B, L, H, D/H) layout for Q/K -- the fused RoPE
        # kernel expects (batch, seqlen, nheads, headdim), i.e. seqlen BEFORE nheads,
        # which is the layout produced by .view() before the SDPA .transpose(1, 2).
        q = self.q_proj(normed).view(B, L, self.n_heads, self.head_dim)                 # [B, L, H, D/H]
        k = self.k_proj(normed).view(B, L, self.n_heads, self.head_dim)                 # [B, L, H, D/H]
        v = self.v_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, D/H]

        # Rotary position embedding -- rotate Q and K only (V is untouched), then
        # transpose into SDPA's expected [B, H, L, D/H] layout. inplace=True is safe:
        # q/k are fresh tensors from independent q_proj/k_proj outputs, not aliased
        # views into a shared packed-QKV tensor.
        q = apply_rotary_emb_func(q, cos, sin, False, True).transpose(1, 2)  # [B, H, L, D/H]
        k = apply_rotary_emb_func(k, cos, sin, False, True).transpose(1, 2)  # [B, H, L, D/H]

        # Flash attention (memory-efficient, uses fused kernels). attn_mask broadcasts
        # [B, 1, 1, L] over heads and query positions -- masking out padded keys here
        # is mathematically identical to running the cropped (unpadded) sequence alone.
        attn_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
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

    Input: [B, 1500, 768] noisy latent + timestep t
    Output: [B, 1500, 768] predicted noise (ε-prediction)

    With cond_acc=True (I²SB cond_acc): z_acc is concatenated to z_t along the
    feature dimension before proj_in, giving the model a fixed anchor to the
    source accent throughout the denoising trajectory.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        latent_dim: int = 768,
        cond_acc: bool = False,
        parameterization: str = "eps",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.cond_acc = cond_acc
        self.parameterization = parameterization

        # Input projection: (latent_dim * 2 if cond_acc else latent_dim) -> d_model
        in_dim = latent_dim * 2 if cond_acc else latent_dim
        self.proj_in = nn.Linear(in_dim, d_model)

        # RoPE cache — built once for the fixed 1500-frame buffer length every
        # encoder state is stored at (BridgeDataset asserts z_acc/z_nat are always
        # [1500, 768]). Full rotary (n_elem == head_dim), matching the dominant
        # convention elsewhere in this repo's transformer configs. Registered as
        # non-persistent buffers: they move/cast automatically with model.to(device,
        # dtype) and are excluded from checkpoint state_dicts (no learnable
        # parameters -> no key-mismatch risk when loading old checkpoints).
        head_dim = d_model // n_heads
        rope_cos, rope_sin = build_rope_cache(
            seq_len=1500, n_elem=head_dim, dtype=torch.float32, device=torch.device("cpu")
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)

        # Stack of AdaLN blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)

        # Output projection: d_model -> latent_dim
        self.proj_out = nn.Linear(d_model, latent_dim)

        # Zero-initialize output projection for both parameterizations.
        # eps: eps_pred=0 → z_nat_hat=z_t → identity at init.
        # x0: proj_out=0 + z_t residual (see forward) → z_nat_hat=z_t → identity at init.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                z_acc: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t:   [B, L, 768] noisy latent at timestep t
            t:     [B] timesteps in [0, 1]
            z_acc: [B, L, 768] accented encoder states — required if cond_acc=True,
                   ignored otherwise (I²SB cond_acc)
            key_padding_mask: optional [B, L] bool, True = real frame, False = padding —
                   forwarded unchanged to every block's attention (see AdaLNBlock.forward)

        Returns:
            eps_pred: [B, L, 768] predicted noise
        """
        if self.cond_acc:
            assert z_acc is not None, "z_acc required when cond_acc=True"
            x = torch.cat([z_t, z_acc.to(z_t.dtype)], dim=-1)  # [B, L, 1536]
        else:
            x = z_t  # [B, L, 768]

        x = self.proj_in(x)              # [B, L, d_model]
        t_emb = self.time_embedding(t)   # [B, d_model]

        # Slice the precomputed RoPE tables to this batch's actual sequence length --
        # max_len varies per batch (max(speech_end) + 1 <= 1500), it is not always
        # the full 1500-frame buffer.
        L = z_t.shape[1]
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]

        for block in self.blocks:
            x = block(x, t_emb, cos, sin, key_padding_mask=key_padding_mask)

        x = self.norm_final(x)
        out = self.proj_out(x)           # [B, L, 768]
        if self.parameterization == "x0":
            out = out + z_t              # residual: predict correction on z_t, not absolute z_nat
        return out
