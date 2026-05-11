# -*- coding: utf-8 -*-
"""
Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class TransEncoder(nn.Module):
   def __init__(self, config: Config) -> None:
       super().__init__()
       assert config.padded_vocab_size is not None
       self.config = config

       self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

       self.transformer = nn.ModuleDict(
           dict(
               wte=nn.Embedding(
                   config.padded_vocab_size + 1, config.n_embd
               ),  # +1 for [MASK] token
               h=nn.ModuleList(
                   Block(config) for _ in range(config.n_layer)
               ),
               ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
           )
       )

       self.rope_cache: Optional[RoPECache] = None

   def _init_weights(self, module: nn.Module, n_layer) -> None:
       """Initialize weights following GPT-NeoX paper"""
       if isinstance(module, nn.Embedding):
           torch.nn.init.normal_(
               module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
           )
       elif isinstance(module, nn.Linear):
           torch.nn.init.normal_(
               module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
           )
           if module.bias is not None:
               torch.nn.init.zeros_(module.bias)
       
       for name, p in module.named_parameters():
           if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (
               name == "w3.weight"
               and isinstance(module, SwiGLU)
               or (name == "proj.weight" and isinstance(module, SelfAttention))
           ):
               nn.init.normal_(
                   p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer
               )

   def forward(
       self, 
       idx: torch.Tensor,
       condition: Optional[torch.Tensor] = None  # Whisfusion: audio condition
   ) -> torch.Tensor:
       B, T = idx.size()
       
       block_size = self.config.block_size
       assert (
           block_size >= T
       ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

       if self.rope_cache is None:
           model_dtype = next(self.parameters()).dtype
           self.rope_cache = self.build_rope_cache(idx, dtype=model_dtype, device=idx.device)

       cos, sin = self.rope_cache
       cos = cos[:T]
       sin = sin[:T]

       x = self.transformer.wte(idx)

       # Pass condition to each block for cross-attention
       for block in self.transformer.h:
           x = block(x, (cos, sin), condition=condition)

       x = self.transformer.ln_f(x)

       return self.lm_head(x)

   @classmethod
   def from_name(cls, name: str, **kwargs: Any) -> Self:
       return cls(Config.from_name(name, **kwargs))

   def build_rope_cache(self, idx: torch.Tensor, dtype: torch.dtype, device: torch.device) -> RoPECache:
       return build_rope_cache(
           seq_len=self.config.block_size,
           n_elem=int(self.config.rotary_percentage * self.config.head_size),
           dtype=dtype,
           device=device,
           condense_ratio=self.config.condense_ratio,
       )


class Block(nn.Module):
   def __init__(self, config: Config) -> None:
       super().__init__()
       self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
       self.attn = SelfAttention(config)
       
       # Whisfusion: Cross-attention components
       self.norm_cross = config.norm_class(config.n_embd, eps=config.norm_eps)
       self.cross_attn = CrossAttention(config)
       
       if not config.shared_attention_norm:
           self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
       self.mlp = config.mlp_class(config)
       self.config = config

   def forward(
       self,
       x: torch.Tensor,
       rope: RoPECache,
       condition: Optional[torch.Tensor] = None
   ) -> torch.Tensor:
       
       # Self-attention
       h = self.attn(self.norm_1(x), rope)
       x = x + h
       
       # Cross-attention (only when condition is provided)
       if condition is not None:
           h_cross = self.cross_attn(self.norm_cross(x), condition)
           x = x + h_cross

       # MLP
       x = x + self.mlp(self.norm_2(x))
       
       return x


class SelfAttention(nn.Module):
   def __init__(self, config: Config) -> None:
       super().__init__()

       # Combined QKV projection for efficiency
       shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
       self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)

       self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

       self.config = config

   def forward(
       self,
       x: torch.Tensor,
       rope: RoPECache,
   ) -> Tuple[torch.Tensor, Optional[KVCache]]:
       B, T, C = x.size()

       qkv = self.attn(x)

       # Support for MHA, MQA and GQA
       q_per_kv = self.config.n_head // self.config.n_query_groups
       total_qkv = q_per_kv + 2
       qkv = qkv.view(
           B, T, self.config.n_query_groups, total_qkv, self.config.head_size
       )

       q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

       q = q.reshape(B, T, -1, self.config.head_size)
       k = k.reshape(B, T, -1, self.config.head_size)
       v = v.reshape(B, T, -1, self.config.head_size)

       cos, sin = rope

       # Apply RoPE in fp32 for stability
       q = apply_rotary_emb_func(q, cos, sin, False, True)
       k = apply_rotary_emb_func(k, cos, sin, False, True)

       y = self.scaled_dot_product_attention(q, k, v)

       y = y.reshape(B, T, C)

       y = self.proj(y)

       return y

   def scaled_dot_product_attention(
       self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
   ):
       scale = 1.0 / math.sqrt(self.config.head_size)

       if (
           FlashAttention2Available
           and q.device.type == "cuda"
           and q.dtype in (torch.float16, torch.bfloat16)
       ):
           from flash_attn import flash_attn_func

           return flash_attn_func(
               q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False
           )
       q = q.transpose(1, 2)
       k = k.transpose(1, 2)
       v = v.transpose(1, 2)
       if q.size() != k.size():
           k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
           v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
       y = torch.nn.functional.scaled_dot_product_attention(
           q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False
       )
       return y.transpose(1, 2)


class CrossAttention(nn.Module):
   """Cross-attention module where Q comes from decoder (text) and K,V from encoder (audio)"""
   def __init__(self, config: Config) -> None:
       super().__init__()
       self.config = config
       
       self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
       self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
       self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

   def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
       B, T, C = x.size()
       _, T_cond, _ = condition.size()

       q = self.q_proj(x)
       k, v = self.kv_proj(condition).split(self.config.n_embd, dim=-1)

       q = q.view(B, T, self.config.n_head, self.config.head_size)
       k = k.view(B, T_cond, self.config.n_head, self.config.head_size)
       v = v.view(B, T_cond, self.config.n_head, self.config.head_size)

       q = q.transpose(1, 2)
       k = k.transpose(1, 2)
       v = v.transpose(1, 2)
       
       y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
       
       y = y.transpose(1, 2).contiguous().view(B, T, C)
       
       return self.proj(y)


class GptNeoxMLP(nn.Module):
   def __init__(self, config: Config) -> None:
       super().__init__()
       self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
       self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = self.fc(x)
       x = torch.nn.functional.gelu(x)
       return self.proj(x)


class LLaMAMLP(nn.Module):
   def __init__(self, config: Config) -> None:
       super().__init__()
       self.swiglu = SwiGLU(
           config.n_embd, config.intermediate_size, bias=False, _pack_weights=False
       )

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.swiglu(x)


def build_rope_cache(
   seq_len: int,
   n_elem: int,
   dtype: torch.dtype,
   device: torch.device,
   base: int = 10000,
   condense_ratio: int = 1,
) -> RoPECache:
   """Enhanced Transformer with Rotary Position Embedding."""
   theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device, dtype=dtype) / n_elem))
   seq_idx = torch.arange(seq_len, device=device, dtype=dtype) / condense_ratio
   idx_theta = torch.outer(seq_idx, theta)
   
   cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
   
   return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
   head_size = x.size(-1)
   x1 = x[..., : head_size // 2]
   x2 = x[..., head_size // 2 :]
   rotated = torch.cat((-x2, x1), dim=-1)
   roped = (x * cos) + (rotated * sin)
   return roped.type_as(x)