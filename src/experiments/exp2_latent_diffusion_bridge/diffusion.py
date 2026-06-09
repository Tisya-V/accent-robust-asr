"""
I²SB latent diffusion bridge.
Based on I²SB (Liu et al., 2023): https://arxiv.org/abs/2302.05872

Forward process:
  z_t = z_t_clean(t) + σ_bridge(t)·ε·speech_mask
  z_t_clean(t) = (1-t)·z_nat + t·z_acc          (bridge mean — both endpoints known)
  σ_bridge(t) = sigma_max·√(t(1-t))            peaks at t=0.5, zero at endpoints

Three parameterizations (PARAM_REGISTRY):
  eps  — I2SB-style: target = (z_t - z_nat) / (σ_max·√t)
         = √t·(z_acc-z_nat)/σ_max + √(1-t)·ε  [bounded at both endpoints]
         recovery: z_nat_hat = z_t - σ_max·√t·net_out  (no 1/(1-t))
  x0   — predicts z_nat_canon directly
         recovery: z_nat_hat = model output
  cfm  — predicts velocity v = dz_t/dt
         inference: Euler step z_{t-dt} = z_t - v·dt (no x0-blend)

Alignment modes:
  position  — frame-by-frame blend; bridge_loss()
  dtw       — DTW alpha-timeline blend with N(t) morphing; bridge_loss_dtw()
  dtw_fixed — fixed DTW mapping on L2 timeline; no N(t) morphing; bridge_loss_dtw()
              Each L2 frame k maps to nat_idx[k] via DTW, bridge runs in T_l2 frame space.
              Dispatched via _DTW_DISPATCH in bridge_loss_dtw().

Special case (also via _DTW_DISPATCH):
  cfm_prewarp — DTW-only; freezes alignment at t=1, straight-line blend on L2 timeline,
                CFM velocity parameterization. Dispatched by parameterization key.

Inference ODE step (shared across eps/x0, parameterization-independent):
  z_{t'} = (1 - t'/t)·z_nat_hat + (t'/t)·z_t

For dtw_fixed inference: pass T_nat=T_l2 to bridge_inference — this collapses N(t) to a
constant so no frame-count morphing occurs during the reverse ODE.

sigma_max calibration: per-element std of (z_acc - z_nat) over speech frames.
Measured on hindi-only train split: 1.535 → use sigma_max=1.5.
See claude-files/sigma_max_calibration.md for calibration code and rationale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def sigma_bridge(t: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """σ_bridge(t) = sigma_max·√(t(1-t))  — zero at both endpoints."""
    return sigma_max * torch.sqrt((t * (1 - t)).clamp(min=1e-5))


# ---------------------------------------------------------------------------
# Shared forward-process helpers
# ---------------------------------------------------------------------------

def _build_z_t_clean_position(
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    t_batch: torch.Tensor,
    speech_mask: torch.Tensor,
    pos: torch.Tensor,
    N_batch: torch.Tensor,
    l2_speech_end: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Bridge mean for position-aligned mode.

    N = max(T_l2, T_nat) is fixed (no per-step morphing — see `bridge_loss`).
    Speech frames [0:N]: linear blend (1-t)·z_nat[i] + t·z_acc[i].
    Tail frames [N:]:    z_acc[T_l2 + offset] — on-manifold L2 silence.
    """
    B, _, D = z_nat.shape
    t_view         = t_batch.view(B, 1, 1)
    z_speech_blend = (1 - t_view) * z_nat + t_view * z_acc                                  # [B, max_len, D]
    tail_offset    = (pos - N_batch.unsqueeze(1)).clamp(min=0)                               # [B, max_len]
    tail_src       = (l2_speech_end.long().unsqueeze(1) + tail_offset).clamp(0, max_len - 1)
    z_tail         = torch.gather(z_acc, 1, tail_src.unsqueeze(-1).expand(-1, -1, D))
    return torch.where(speech_mask.unsqueeze(-1), z_speech_blend, z_tail)                   # [B, max_len, D]


def _build_z_t_clean_dtw(
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    t_batch: torch.Tensor,
    path_tensor: torch.Tensor,
    T_l2: torch.Tensor,
    T_nat: torch.Tensor,
    N_batch: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bridge mean for DTW-aligned mode.

    Speech frames [0:N(t)]: DTW alpha-timeline blend — nearest-neighbour lookup on
      t_k = t·l2_norm + (1-t)·nat_norm, then (1-t)·z_nat[nat_idx] + t·z_acc[l2_idx].
    Tail frames [N(t):]:    L2 (on-manifold accented) silence.

    Returns z_t_clean and intermediate tensors needed by CFM velocity target:
      (z_t_clean, speech_acc, speech_nat, speech_pos, speech_mask)
    """
    B, _, D = z_nat.shape
    device  = z_nat.device
    max_P   = path_tensor.shape[1]
    max_N   = int(N_batch.max())
    t_view  = t_batch.view(B, 1, 1)

    # Blended path timeline
    nat_norm = path_tensor[:, :, 0].float() / (T_nat - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    l2_norm  = path_tensor[:, :, 1].float() / (T_l2  - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    t_k      = t_batch[:, None] * l2_norm + (1 - t_batch[:, None]) * nat_norm         # [B, max_P]

    # Output grid: k / (N_b - 1) for each item
    k_grid = torch.arange(max_N, device=device, dtype=torch.float32).unsqueeze(0)
    out_t  = (k_grid / (N_batch - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)  # [B, max_N]

    # Nearest-neighbour lookup on blended timeline
    idx_r = torch.searchsorted(t_k.contiguous(), out_t.contiguous()).clamp(0, max_P - 1)
    idx_l = (idx_r - 1).clamp(0, max_P - 1)
    t_k_r = torch.gather(t_k, 1, idx_r)
    t_k_l = torch.gather(t_k, 1, idx_l)
    k_idx = torch.where((t_k_l - out_t).abs() <= (t_k_r - out_t).abs(), idx_l, idx_r)  # [B, max_N]

    nat_idx    = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx)  # [B, max_N]
    l2_idx     = torch.gather(path_tensor[:, :, 1].long(), 1, k_idx)  # [B, max_N]
    speech_acc = torch.gather(z_acc, 1, l2_idx.unsqueeze(-1).expand(-1, -1, D))   # [B, max_N, D]
    speech_nat = torch.gather(z_nat, 1, nat_idx.unsqueeze(-1).expand(-1, -1, D))  # [B, max_N, D]
    speech     = (1 - t_view) * speech_nat + t_view * speech_acc                   # [B, max_N, D]

    # Tail silence frames — always L2 (on-manifold accented) silence
    max_need   = max_len - int(N_batch.min())
    tail_range = torch.arange(max_need, device=device).unsqueeze(0)
    tail_i_acc = (T_l2.long().unsqueeze(1) + tail_range).clamp(0, max_len - 1)
    tail       = torch.gather(z_acc, 1, tail_i_acc.unsqueeze(-1).expand(-1, -1, D))

    # Assemble full [B, max_len, D]
    pos         = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
    speech_mask = pos < N_batch.unsqueeze(1)                                           # [B, max_len]
    speech_pos  = pos.clamp(0, max_N - 1)
    speech_out  = torch.gather(speech, 1, speech_pos.unsqueeze(-1).expand(-1, -1, D)) * speech_mask.unsqueeze(-1)
    tail_pos    = (pos - N_batch.unsqueeze(1)).clamp(0, max_need - 1)
    tail_out    = torch.gather(tail,   1, tail_pos.unsqueeze(-1).expand(-1, -1, D))   * (~speech_mask).unsqueeze(-1)
    z_t_clean     = speech_out + tail_out                                                 # [B, max_len, D]

    return z_t_clean, speech_acc, speech_nat, speech_pos, speech_mask


def _apply_bridge_noise(
    z_t_clean: torch.Tensor,
    sigma_br: torch.Tensor,
    speech_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add bridge noise to speech frames only. Returns (z_t, eps)."""
    eps = torch.randn_like(z_t_clean)
    z_t = z_t_clean + sigma_br * eps * speech_mask.unsqueeze(-1)
    return z_t, eps


def _build_endpoint(
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    nat_speech_end: torch.Tensor,
    l2_speech_end: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build z_nat_canon and tail_mask.

    z_nat_canon = z_nat[:T_nat] || z_acc[T_l2 + offset] at [T_nat:]
    tail_mask   = frames at or beyond T_nat (the native speech boundary)

    z_nat_canon is the t=0 endpoint target; tail_mask separates speech from silence
    in the loss weighting.
    """
    B, _, D = z_nat.shape
    device  = z_nat.device
    pos         = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)      # [B, max_len]
    tail_mask   = nat_speech_end.unsqueeze(1) <= pos                                    # [B, max_len]
    tail_offset = pos - nat_speech_end.unsqueeze(1)                                     # [B, max_len]
    tail_src    = (l2_speech_end.long().unsqueeze(1) + tail_offset).clamp(0, max_len - 1)
    z_acc_tail  = torch.gather(z_acc, 1, tail_src.unsqueeze(-1).expand(-1, -1, D))
    z_nat_canon = torch.where(tail_mask.unsqueeze(-1), z_acc_tail, z_nat)               # [B, max_len, D]
    return z_nat_canon, tail_mask


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sample: bool,
) -> tuple[torch.Tensor, ...]:
    """MSE over `mask` frames only — tail frames (always on-manifold L2 silence,
    already visible to the model via cond_acc) are excluded; nothing to learn there.

    per_sample=True  → returns ([B],) tensor (no backward; used for min-over-natives val).
    per_sample=False → returns (loss, loss.item()).
    """
    diff_sq = (pred - target).pow(2)
    mask_f  = mask.float().unsqueeze(-1)
    D       = pred.shape[-1]

    if per_sample:
        per = (diff_sq * mask_f).sum(dim=(1, 2)) / (mask_f.sum(dim=(1, 2)).clamp(min=1) * D)
        return (per,)

    loss = (diff_sq * mask_f).sum() / (mask_f.sum().clamp(min=1) * D)
    return loss, loss.item()


# ---------------------------------------------------------------------------
# Parameterization classes
# ---------------------------------------------------------------------------

class BridgeParameterization(ABC):
    """Base class for bridge parameterizations.

    Subclasses implement:
      compute_target — returns the training label given forward-process tensors.
      recover_x0     — recovers z_nat_hat from model output at inference.
    """

    @abstractmethod
    def compute_target(
        self,
        z_t: torch.Tensor,
        z_t_clean: torch.Tensor,
        sigma_br: torch.Tensor,
        z_nat_canon: torch.Tensor,
        velocity: torch.Tensor | None = None,
        sigma_fwd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def recover_x0(
        self,
        z_t: torch.Tensor,
        z_acc: torch.Tensor,
        t_cur: torch.Tensor,
        model_out: torch.Tensor,
        sigma_br: torch.Tensor,
        sigma_fwd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


class EpsParam(BridgeParameterization):
    """I2SB-style ε-prediction — normalises by σ_fwd = σ_max·√t.

    σ_bridge(t) = σ_max·√(t(1-t)) collapses at both endpoints, which forces the
    naive recovery to divide by (1-t) → unstable near t=1. Instead, following
    I2SB (Liu et al., 2023) we normalise by the one-sided forward std σ_fwd = σ_max·√t,
    which is monotone and never zero for t > 0.

    Forward algebra:
      z_t - z_nat = t·(z_acc - z_nat) + σ_max·√(t(1-t))·ε

    Divide by σ_fwd = σ_max·√t:
      target = √t·(z_acc - z_nat)/σ_max  +  √(1-t)·ε
             → ε                    as t→0  (bounded, unit variance)
             → (z_acc - z_nat)/σ_max  as t→1  (bounded by data)

    Recovery:
      z_nat_hat = z_t - σ_max·√t · net_out
    — exact inversion, no division by (1-t).
    """

    def compute_target(self, z_t, z_t_clean, sigma_br, z_nat_canon, velocity=None, sigma_fwd=None):
        denom = sigma_fwd if sigma_fwd is not None else sigma_br
        return (z_t - z_nat_canon) / (denom + 1e-8)

    def recover_x0(self, z_t, z_acc, t_cur, model_out, sigma_br, sigma_fwd=None):
        denom = sigma_fwd if sigma_fwd is not None else sigma_br
        return z_t - denom * model_out


class X0Param(BridgeParameterization):
    """Direct endpoint prediction.

    target   = z_nat_canon
    recovery = model output (already z_nat_hat)
    """

    def compute_target(self, z_t, z_t_clean, sigma_br, z_nat_canon, velocity=None, sigma_fwd=None):
        return z_nat_canon

    def recover_x0(self, z_t, z_acc, t_cur, model_out, sigma_br, sigma_fwd=None):
        return model_out


class CFMParam(BridgeParameterization):
    """Continuous flow matching — predicts velocity v = dz_t/dt.

    target   = pre-computed velocity tensor (passed as `velocity` kwarg)
    inference: Euler step z_{t-dt} = z_t - v·dt  (no x0-blend;
      recover_x0 unused)
    """

    def compute_target(self, z_t, z_t_clean, sigma_br, z_nat_canon=None, velocity=None, sigma_fwd=None):
        if velocity is None:
            raise ValueError("CFMParam.compute_target requires velocity kwarg")
        return velocity

    def recover_x0(self, z_t, z_acc, t_cur, model_out, sigma_br, sigma_fwd=None):
        raise NotImplementedError("CFM uses Euler steps, not x0 recovery")


PARAM_REGISTRY: dict[str, BridgeParameterization] = {
    "eps": EpsParam(),
    "x0":  X0Param(),
    "cfm": CFMParam(),
}


# ---------------------------------------------------------------------------
# cfm_prewarp special case (DTW-only)
# ---------------------------------------------------------------------------

def _bridge_loss_dtw_cfm_prewarp(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    t_batch: torch.Tensor,
    l2_norm: torch.Tensor,
    path_tensor: torch.Tensor,
    T_l2: torch.Tensor,
    sigma_max: float,
    parameterization: str,  # unused — cfm_prewarp always predicts velocity
    per_sample: bool,
    pos: torch.Tensor,
    max_len: int,
    return_x0: bool = False,
) -> tuple:
    """CFM-prewarp training loss (DTW-only).

    Freezes DTW alignment at t=1: the L2 timeline is fixed, and each L2 frame k
    is mapped to its DTW-matched native frame nat_idx_pw[k]. Training uses a
    straight-line blend on the L2 timeline — no N(t) morphing.

    Velocity target: z_acc[k] - z_nat_warped[k] per L2 speech frame.
    """
    B, _, D = z_nat.shape
    device  = z_nat.device
    max_P   = path_tensor.shape[1]
    t_view  = t_batch.view(B, 1, 1)
    max_l2  = int(T_l2.max())

    # Map each L2 frame k → its DTW-matched native frame (fixed at t=1 alignment)
    k_grid_pw = torch.arange(max_l2, device=device, dtype=torch.float32).unsqueeze(0)
    out_t_pw  = (k_grid_pw / (T_l2 - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)
    idx_r_pw  = torch.searchsorted(l2_norm.contiguous(), out_t_pw.contiguous()).clamp(0, max_P - 1)
    idx_l_pw  = (idx_r_pw - 1).clamp(0, max_P - 1)
    k_idx_pw  = torch.where(
        (torch.gather(l2_norm, 1, idx_l_pw) - out_t_pw).abs()
        <= (torch.gather(l2_norm, 1, idx_r_pw) - out_t_pw).abs(),
        idx_l_pw, idx_r_pw,
    )
    nat_idx_pw   = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx_pw)  # [B, max_l2]
    z_nat_warped = torch.gather(
        z_nat, 1, nat_idx_pw.clamp(0, max_len - 1).unsqueeze(-1).expand(-1, -1, D),
    )  # [B, max_l2, D]

    # Straight-line forward process on L2 timeline; tail = z_acc silence
    z_acc_l2   = z_acc[:, :max_l2, :]
    z_t_clean_pw = (1 - t_view) * z_nat_warped + t_view * z_acc_l2
    if sigma_max > 0:
        sigma_br_pw = sigma_bridge(t_batch, sigma_max).view(B, 1, 1)
        z_t_clean_pw  = z_t_clean_pw + sigma_br_pw * torch.randn_like(z_t_clean_pw)
    z_t_pw               = z_acc.clone()
    z_t_pw[:, :max_l2, :] = z_t_clean_pw

    # Speech mask: fixed at pos < T_l2 (no N(t) morphing)
    sm_pw = pos < T_l2.long().unsqueeze(1)  # [B, max_len]

    # Velocity target: z_acc[k] - z_nat_warped[k], zero outside L2 speech
    v_tgt                 = torch.zeros_like(z_acc)
    v_tgt[:, :max_l2, :] = (z_acc_l2 - z_nat_warped) * sm_pw[:, :max_l2].unsqueeze(-1)

    pred = model(z_t_pw, t_batch, z_acc)
    if per_sample:
        return _masked_mse(pred, v_tgt, sm_pw, per_sample=True)
    loss, loss_val = _masked_mse(pred, v_tgt, sm_pw, per_sample=False)
    return loss, loss_val, 0.0, None  # x0_hat N/A for cfm_prewarp (Euler parameterization)


def _bridge_loss_dtw_fixed(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    t_batch: torch.Tensor,
    l2_norm: torch.Tensor,
    path_tensor: torch.Tensor,
    T_l2: torch.Tensor,
    sigma_max: float,
    parameterization: str,
    per_sample: bool,
    pos: torch.Tensor,
    max_len: int,
    return_x0: bool = False,
) -> tuple:
    """Fixed-timeline DTW bridge loss (eps / x0).

    Freezes the DTW alignment on the L2 timeline: each L2 frame k maps to
    nat_idx[k] via nearest-neighbour on j_norm. The bridge runs in [B, T_l2, D]
    space with no N(t) morphing. Frames [T_l2:] are held at z_acc silence.

      z_0 endpoint: z_nat[nat_idx[k]]  (DTW-warped native)
      z_1 endpoint: z_acc[k]

    Loss is MSE over L2 speech frames only.
    """
    B, _, D = z_nat.shape
    device  = z_nat.device
    max_P   = path_tensor.shape[1]
    t_view  = t_batch.view(B, 1, 1)
    max_l2  = int(T_l2.max())

    # Project path onto L2 grid: for each L2 frame k find its DTW-matched native frame
    k_grid = torch.arange(max_l2, device=device, dtype=torch.float32).unsqueeze(0)
    out_t  = (k_grid / (T_l2 - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)
    idx_r  = torch.searchsorted(l2_norm.contiguous(), out_t.contiguous()).clamp(0, max_P - 1)
    idx_l  = (idx_r - 1).clamp(0, max_P - 1)
    k_idx  = torch.where(
        (torch.gather(l2_norm, 1, idx_l) - out_t).abs()
        <= (torch.gather(l2_norm, 1, idx_r) - out_t).abs(),
        idx_l, idx_r,
    )
    nat_idx      = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx)         # [B, max_l2]
    z_nat_warped = torch.gather(
        z_nat, 1, nat_idx.clamp(0, max_len - 1).unsqueeze(-1).expand(-1, -1, D),
    )                                                                            # [B, max_l2, D]

    # Forward bridge on L2 timeline
    z_acc_l2  = z_acc[:, :max_l2, :]
    z_t_clean = (1 - t_view) * z_nat_warped + t_view * z_acc_l2
    sigma_br  = sigma_bridge(t_batch, sigma_max).view(B, 1, 1)
    sigma_fwd = (sigma_max * t_batch.sqrt()).view(B, 1, 1)
    sm        = pos[:, :max_l2] < T_l2.long().unsqueeze(1)                     # [B, max_l2]
    eps       = torch.randn_like(z_t_clean)
    z_t_l2    = z_t_clean + sigma_br * eps * sm.unsqueeze(-1)

    # Full-length input for model: bridge frames + z_acc silence beyond T_l2
    z_t_full                 = z_acc.clone()
    z_t_full[:, :max_l2, :] = z_t_l2

    pred   = model(z_t_full, t_batch, z_acc)[:, :max_l2, :]                    # [B, max_l2, D]
    target = PARAM_REGISTRY[parameterization].compute_target(
        z_t_l2, z_t_clean, sigma_br, z_nat_warped, sigma_fwd=sigma_fwd,
    )

    if per_sample:
        return _masked_mse(pred, target, sm, per_sample=True)
    loss, loss_val = _masked_mse(pred, target, sm, per_sample=False)
    if return_x0:
        # eps: recover z_nat_hat = z_t - sigma_fwd * pred (exact inversion of EpsParam target)
        # x0:  pred already IS z_nat_hat (model.forward adds z_t residual)
        x0_hat = pred if parameterization == "x0" else z_t_l2 - sigma_fwd * pred
    else:
        x0_hat = None
    return loss, loss_val, 0.0, x0_hat


# Registry for DTW special-case variants that bypass the N(t) alpha-timeline path.
# Key is either the parameterization value (cfm_prewarp) or the alignment value (dtw_fixed).
# bridge_loss_dtw checks parameterization first, then alignment.
_DTW_DISPATCH: dict[str, object] = {
    "cfm_prewarp": _bridge_loss_dtw_cfm_prewarp,
    "dtw_fixed":   _bridge_loss_dtw_fixed,
}


# ---------------------------------------------------------------------------
# Training losses
# ---------------------------------------------------------------------------

def bridge_loss(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_end: torch.Tensor,
    nat_speech_end: torch.Tensor,
    sigma_max: float = 0.5,
    parameterization: str = "eps",
    per_sample: bool = False,
) -> tuple:
    """Training loss — position-aligned bridge.

    Active speech region is fixed at N = max(T_l2, T_nat) — no per-step morphing,
    since position alignment is always identity correspondence (z_nat[i] <-> z_acc[i])
    and has no second alignment to morph between.
    Speech [0:N]: frame-by-frame blend. Tail [N:]: z_acc silence.
    Loss: masked MSE over canonical speech frames [0:T_nat] only — tail frames
    (always on-manifold L2 silence) carry nothing to learn and are excluded.
    """
    B, _, D = z_nat.shape
    device  = z_nat.device

    max_len = int(max(l2_speech_end.max(), nat_speech_end.max()).item()) + 1
    z_nat   = z_nat[:, :max_len, :]
    z_acc   = z_acc[:, :max_len, :]

    t_batch = torch.rand(B, device=device, dtype=torch.float32)
    N_batch = torch.maximum(l2_speech_end, nat_speech_end).long().clamp(min=1)

    pos         = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
    speech_mask = pos < N_batch.unsqueeze(1)

    z_t_clean  = _build_z_t_clean_position(z_nat, z_acc, t_batch, speech_mask, pos, N_batch, l2_speech_end, max_len)
    sigma_br  = sigma_bridge(t_batch, sigma_max).view(B, 1, 1)
    sigma_fwd = (sigma_max * t_batch.sqrt()).view(B, 1, 1)
    z_t, _    = _apply_bridge_noise(z_t_clean, sigma_br, speech_mask)

    z_nat_canon, tail_mask = _build_endpoint(z_nat, z_acc, nat_speech_end, l2_speech_end, max_len)

    sp_mask = speech_mask & ~tail_mask

    # CFM velocity: (z_acc[i] - z_nat[i]) on speech-only frames
    velocity = None
    if parameterization == "cfm":
        velocity = (z_acc - z_nat) * sp_mask.unsqueeze(-1)

    param  = PARAM_REGISTRY[parameterization]
    pred   = model(z_t, t_batch, z_acc)
    target = param.compute_target(z_t, z_t_clean, sigma_br, z_nat_canon, velocity=velocity, sigma_fwd=sigma_fwd)

    if per_sample:
        return _masked_mse(pred, target, sp_mask, per_sample=True)
    loss, loss_val = _masked_mse(pred, target, sp_mask, per_sample=False)
    return loss, loss_val, 0.0


def bridge_loss_dtw(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_ends: torch.Tensor,
    nat_speech_ends: torch.Tensor,
    path_tensor: torch.Tensor,
    sigma_max: float = 0.5,
    parameterization: str = "eps",
    alignment: str = "dtw",
    lambda_v: float = 0.0,
    per_sample: bool = False,
    return_x0: bool = False,
) -> tuple:
    """Training loss — DTW-aligned bridge.

    Speech frames use the DTW alpha-timeline interpolation; tail frames use L2
    (on-manifold accented) silence and are excluded from the loss (nothing to
    learn there — the model already sees this content via cond_acc). Special-case
    variants in _DTW_DISPATCH are dispatched before building the shared
    forward-process state.

    path_tensor[:, :, 0] = native indices, [:, :, 1] = L2 indices.
    Paths padded to [B, max_P, 2] in collate_fn_dtw (last point repeated).
    """
    B, _, D = z_nat.shape
    device  = z_nat.device

    # Crop — reduces attention from O(1500²) to O(max_len²); p99 max_len ≈ 380
    max_len = int(max(l2_speech_ends.max(), nat_speech_ends.max()).item()) + 1
    z_nat   = z_nat[:, :max_len, :]
    z_acc   = z_acc[:, :max_len, :]

    T_l2  = l2_speech_ends.float()
    T_nat = nat_speech_ends.float()

    t_batch = torch.rand(B, device=device, dtype=torch.float32)
    N_batch = (t_batch * T_l2 + (1 - t_batch) * T_nat).round().long().clamp(min=1)

    pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)

    # l2_norm is needed by cfm_prewarp and by _build_z_t_clean_dtw — compute once
    max_P    = path_tensor.shape[1]
    l2_norm  = path_tensor[:, :, 1].float() / (T_l2 - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]

    dispatch_key = parameterization if parameterization in _DTW_DISPATCH else alignment
    if dispatch_key in _DTW_DISPATCH:
        return _DTW_DISPATCH[dispatch_key](
            model, z_nat, z_acc, t_batch, l2_norm, path_tensor,
            T_l2, sigma_max, parameterization, per_sample, pos, max_len,
            return_x0=return_x0,
        )

    z_t_clean, speech_acc, speech_nat, speech_pos, speech_mask = _build_z_t_clean_dtw(
        z_nat, z_acc, t_batch, path_tensor, T_l2, T_nat, N_batch, max_len,
    )
    sigma_br  = sigma_bridge(t_batch, sigma_max).view(B, 1, 1)
    sigma_fwd = (sigma_max * t_batch.sqrt()).view(B, 1, 1)
    z_t, _    = _apply_bridge_noise(z_t_clean, sigma_br, speech_mask)

    z_nat_canon, tail_mask = _build_endpoint(z_nat, z_acc, nat_speech_ends, l2_speech_ends, max_len)

    # CFM velocity: z_acc[j_k] - z_nat[i_k] at each DTW-matched output position
    velocity = None
    if parameterization == "cfm":
        max_N   = int(N_batch.max())
        v_raw   = speech_acc - speech_nat                                               # [B, max_N, D]
        v_full  = torch.gather(v_raw, 1, speech_pos.unsqueeze(-1).expand(-1, -1, D))   # [B, max_len, D]
        velocity = v_full * speech_mask.unsqueeze(-1)

    param  = PARAM_REGISTRY[parameterization]
    pred   = model(z_t, t_batch, z_acc)
    target = param.compute_target(z_t, z_t_clean, sigma_br, z_nat_canon, velocity=velocity, sigma_fwd=sigma_fwd)

    sp_mask = speech_mask & ~tail_mask
    if per_sample:
        return _masked_mse(pred, target, sp_mask, per_sample=True)
    loss, loss_val = _masked_mse(pred, target, sp_mask, per_sample=False)

    # DTW alpha-timeline path does not expose x0_hat (CE only used with dtw_fixed)
    # DTW-direction auxiliary loss (x0 only, lambda_v > 0)
    # Penalises the angle between the predicted and true correction directions
    # at each DTW-matched frame pair, weighted by ||dir_true|| so frames with
    # large accent deviation dominate near-identical frames.
    if lambda_v > 0.0:
        if parameterization != "x0":
            raise ValueError("lambda_v > 0 requires parameterization='x0'")
        # _build_z_t_clean_dtw doesn't return nat_idx/l2_idx so we re-derive them here.
        # This is a known duplication; a future refactor can return them from the builder.
        max_N     = int(N_batch.max())
        nat_norm_ = path_tensor[:, :, 0].float() / (T_nat - 1).clamp(min=1).unsqueeze(1)
        t_k_      = t_batch[:, None] * l2_norm + (1 - t_batch[:, None]) * nat_norm_
        k_grid_   = torch.arange(max_N, device=device, dtype=torch.float32).unsqueeze(0)
        out_t_    = (k_grid_ / (N_batch - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)
        idx_r_    = torch.searchsorted(t_k_.contiguous(), out_t_.contiguous()).clamp(0, max_P - 1)
        idx_l_    = (idx_r_ - 1).clamp(0, max_P - 1)
        k_idx_    = torch.where(
            (torch.gather(t_k_, 1, idx_l_) - out_t_).abs() <= (torch.gather(t_k_, 1, idx_r_) - out_t_).abs(),
            idx_l_, idx_r_,
        )
        nat_idx_c = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx_).clamp(0, max_len - 1)
        l2_idx_c  = torch.gather(path_tensor[:, :, 1].long(), 1, k_idx_).clamp(0, max_len - 1)

        exp_nat   = nat_idx_c.unsqueeze(-1).expand(-1, -1, D)
        exp_l2    = l2_idx_c.unsqueeze(-1).expand(-1, -1, D)
        x0_at_nat    = torch.gather(pred,  1, exp_nat)
        z_nat_at_nat = torch.gather(z_nat, 1, exp_nat)
        z_acc_at_l2  = torch.gather(z_acc, 1, exp_l2)
        path_speech  = (
            (nat_idx_c < nat_speech_ends.unsqueeze(1)) &
            (l2_idx_c  < l2_speech_ends.unsqueeze(1))
        ).float().unsqueeze(-1)

        dir_pred = x0_at_nat - z_acc_at_l2
        dir_true = z_nat_at_nat - z_acc_at_l2
        weight   = dir_true.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        cos      = F.cosine_similarity(dir_pred, dir_true, dim=-1, eps=1e-8).unsqueeze(-1)
        vel_loss = ((1 - cos) * weight * path_speech).sum() / (weight * path_speech).sum().clamp(min=1e-8)

        return loss + lambda_v * vel_loss, loss_val, vel_loss.item(), None

    return loss, loss_val, 0.0, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    T_l2: int | torch.Tensor,
    T_nat: int | torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 2.0,
    ode_sampling: bool = False,
    parameterization: str = "eps",
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reverse diffusion — shared by position and DTW alignments. Batched: each
    sample gets its own (T_l2, T_nat) and active/silence schedule.

    Active speech region at each step: [0:N(t)] where N(t) = round(t·T_l2 + (1-t)·T_nat),
    computed independently per sample. Frames [N(t):] are held at on-manifold silence
    (z_acc[T_l2:]) throughout.

    Args:
        model:     BridgeTransformer
        z_acc:     [B, 1500, 768] accented encoder states
        T_l2:      accented speech end frame — scalar (broadcasts to all B) or [B] tensor
        T_nat:     predicted native speech end frame — scalar (broadcasts) or [B] tensor
        n_steps:   ODE steps
        sigma_max: must match training value — drives recover_x0 (sigma_br/sigma_fwd),
            which inverts the training-time target definition. Always keep this at the
            trained value, regardless of `ode_sampling`: zeroing it does not "remove
            injected noise", it makes EpsParam.recover_x0 return `z_t - 0*model_out`,
            silently discarding the network's prediction and freezing the trajectory
            at z_acc (a no-op bridge), since z_t starts at z_acc with no noise added.
        ode_sampling: if True, suppress only the injected stochastic term in the
            reverse step (the `noise_std * randn` Langevin/correction term below),
            giving the deterministic probability-flow ODE for these *trained* weights
            — sigma_max-driven recovery is untouched. This is an inference-time
            ablation on a stochastically-trained model (valid per Song et al.: the
            probability-flow ODE shares marginals with the SDE for the same learned
            drift) — NOT I2SB's `ot_ode`, which is a training-time choice that yields
            a differently-trained network (their q_sample is gated by the same flag).
        key_padding_mask: optional [B, 1500] bool, True = real frame / False = padding.
            Required when batching utterances with different `inf_len` (see below) —
            proven equivalent to running each sample at its own length alone (the
            padded-key positions contribute nothing to attention for real positions).
    """
    B, L, D = z_acc.shape
    device, dtype = z_acc.device, z_acc.dtype

    # Broadcast scalar T_l2/T_nat (today's calling convention -- train.py's sanity
    # check, and any single-utterance caller) to [B] tensors with the same value;
    # already-[B] tensors (the new batched eval.py call site) pass through unchanged.
    T_l2  = torch.as_tensor(T_l2,  device=device, dtype=torch.long).expand(B).contiguous()
    T_nat = torch.as_tensor(T_nat, device=device, dtype=torch.long).expand(B).contiguous()

    # Crop to training distribution -- frames beyond max(T_l2, T_nat) are OOD. For a
    # batch this is the largest per-sample boundary; since N(t) is always a convex
    # combination of T_l2 and T_nat, every sample's active region N(t) <= max(T_l2,
    # T_nat) < inf_len, so it always lies inside this shared crop.
    inf_len    = int(torch.maximum(T_l2, T_nat).max().item()) + 1
    z_acc_crop = z_acc[:, :inf_len, :]
    kpm_crop   = key_padding_mask[:, :inf_len] if key_padding_mask is not None else None

    t_schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)
    z_t        = z_acc.clone()

    is_cfm = parameterization in ("cfm", "cfm_prewarp")
    param  = PARAM_REGISTRY.get(parameterization)  # None for cfm_prewarp (uses Euler)

    j = torch.arange(L, device=device)

    def _fill_silence(z: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """z[b, boundary[b]:, :] <- on-manifold L2 silence tiled from z_acc[b, T_l2[b]:].

        Vectorized form of the original scalar
        `z[:, N_t:, :] = z_acc[:, _sil_src(N_t), :]` where
        `_sil_src(N_t) = arange(T_l2, T_l2 + (L - N_t)).clamp(max=L-1)`
        i.e. source index for destination j>=boundary is `clamp(T_l2 + (j - boundary), 0, L-1)`
        -- exactly what src_idx computes below (mirrors the gather+where vectorization
        already used for DTW pre-warp index lookups at train.py:585-594).
        """
        fill    = j.unsqueeze(0) >= boundary.unsqueeze(1)                                       # [B, L]
        src_idx = (T_l2.unsqueeze(1) + (j.unsqueeze(0) - boundary.unsqueeze(1))).clamp(0, L - 1) # [B, L]
        z_sil   = torch.gather(z_acc, 1, src_idx.unsqueeze(-1).expand(-1, -1, D))
        return torch.where(fill.unsqueeze(-1), z_sil, z)

    def _active_boundary(t: float) -> torch.Tensor:
        """N(t) = round(t·T_l2 + (1-t)·T_nat) per sample, clamped to [1, L].

        Computed in float64 to bit-match the original scalar Python arithmetic
        `min(max(1, round(float(t)*T_l2 + (1-float(t))*T_nat)), L)`: `float(t)` is
        already a Python double, Python ints promote to float64 in the multiply, and
        both Python's `round` and `torch.round` use round-half-to-even -- so this
        reproduces the exact same per-sample integer, not an approximation of it.
        """
        t64 = float(t)
        N_t = torch.round(t64 * T_l2.double() + (1.0 - t64) * T_nat.double())
        return N_t.long().clamp(1, L)

    z_t = _fill_silence(z_t, _active_boundary(t_schedule[0].item()))

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t_cur  = t_schedule[i]
            t_next = t_schedule[i + 1]
            t_cur_batch = torch.full((B,), t_cur, device=device, dtype=dtype)

            if is_cfm:
                # Euler step: z_{t-dt} = z_t - v·dt
                v_pred = model(z_t[:, :inf_len, :], t_cur_batch, z_acc_crop, key_padding_mask=kpm_crop)
                dt = (t_cur - t_next).item()
                z_t[:, :inf_len, :] = z_t[:, :inf_len, :] - v_pred * dt
                if parameterization == "cfm":
                    z_t = _fill_silence(z_t, _active_boundary(t_next.item()))  # cfm_prewarp: no mask, region stays at T_l2
                continue

            # x0-recovery path (eps and x0)
            model_out      = model(z_t[:, :inf_len, :], t_cur_batch, z_acc_crop, key_padding_mask=kpm_crop)
            sigma_br_cur   = sigma_bridge(t_cur, sigma_max)
            sigma_fwd_cur  = sigma_max * (t_cur ** 0.5)
            z_nat_hat_crop = param.recover_x0(
                z_t[:, :inf_len, :], z_acc_crop, t_cur, model_out, sigma_br_cur,
                sigma_fwd=sigma_fwd_cur,
            )

            z_nat_hat = z_t.clone()
            z_nat_hat[:, :inf_len, :] = z_nat_hat_crop

            if i == n_steps - 1:
                break

            # ODE step: linear interpolation toward z_nat_hat
            z_t = (1 - t_next / t_cur) * z_nat_hat + (t_next / t_cur) * z_t
            if not ode_sampling:
                # SDE posterior noise: σ(t',t) = sigma_max·√(t'·(t-t')/t)
                # `ode_sampling` suppresses only this injected term -- sigma_max itself
                # stays at the trained value, still driving recover_x0 above (the
                # deterministic probability-flow ODE for these trained weights).
                noise_std = sigma_max * torch.sqrt((t_next * (t_cur - t_next) / t_cur).clamp(min=0.0))
                z_t[:, :inf_len, :] += noise_std * torch.randn(B, inf_len, D, device=device, dtype=dtype)
            z_t = _fill_silence(z_t, _active_boundary(t_next.item()))

    # Final silence enforcement
    if parameterization == "cfm":
        return _fill_silence(z_t, T_nat)
    if parameterization == "cfm_prewarp":
        return _fill_silence(z_t, T_l2)
    return _fill_silence(z_nat_hat, T_nat)


# ---------------------------------------------------------------------------
# Numpy offline forward process (single-sample, diagnostic / preprocessing)
# ---------------------------------------------------------------------------

def _extend_sil(sil: np.ndarray, need: int) -> np.ndarray:
    """Extend a silence slice to exactly `need` frames by tiling the last frame."""
    if len(sil) == 0:
        return np.zeros((need, sil.shape[-1]), dtype=np.float32)
    if len(sil) >= need:
        return sil[:need]
    return np.vstack([sil, np.tile(sil[-1:], (need - len(sil), 1))])


def sample_forward_dtw(
    z_nat_np: np.ndarray,
    z_acc_np: np.ndarray,
    t: float,
    path_arr: np.ndarray,
    T_nat: int,
    T_l2: int,
    eps: np.ndarray,
    sigma_max: float = 0.5,
) -> np.ndarray:
    """DTW-aligned forward process for a single (z_nat, z_acc) pair.

    Numpy equivalent of _build_z_t_clean_dtw + _apply_bridge_noise for offline use.

    Speech frames: DTW alpha-timeline interpolation (N(t) output frames).
    Tail frames: linearly interpolated L2/native silence.
    Bridge noise σ_bridge(t)·ε added to all 1500 frames.

    path_arr convention: path_arr[:, 0] = native indices, [:, 1] = L2 indices.

    Args:
        z_nat_np:  [1500, D] native encoder states
        z_acc_np:  [1500, D] accented encoder states
        t:         timestep in [0, 1]
        path_arr:  [P, 2] int16 DTW warping path
        T_nat:     native speech end frame
        T_l2:      L2 speech end frame
        eps:       [1500, D] pre-generated noise
        sigma_max: noise schedule parameter
    """
    N        = max(1, round(t * T_l2 + (1 - t) * T_nat))
    nat_norm = path_arr[:, 0].astype(np.float32) / max(T_nat - 1, 1)
    l2_norm  = path_arr[:, 1].astype(np.float32) / max(T_l2  - 1, 1)
    t_k      = t * l2_norm + (1 - t) * nat_norm
    out_t    = np.linspace(0.0, 1.0, N, dtype=np.float32)

    idx_r = np.clip(np.searchsorted(t_k, out_t), 0, len(t_k) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(t_k) - 1)
    k_idx = np.where(np.abs(t_k[idx_l] - out_t) <= np.abs(t_k[idx_r] - out_t), idx_l, idx_r)

    nat_idx = path_arr[k_idx, 0].astype(np.int32)
    l2_idx  = path_arr[k_idx, 1].astype(np.int32)
    speech  = ((1 - t) * z_nat_np[nat_idx] + t * z_acc_np[l2_idx]).astype(np.float32)

    need = 1500 - N
    if need > 0:
        l2_pad  = _extend_sil(z_acc_np[T_l2:],  need)
        nat_pad = _extend_sil(z_nat_np[T_nat:], need)
        tail    = ((1 - t) * nat_pad + t * l2_pad).astype(np.float32)
        z_t_clean = np.concatenate([speech, tail], axis=0)
    else:
        z_t_clean = speech[:1500]

    sigma_t = np.float32(sigma_max * np.sqrt(max(t * (1 - t), 1e-5)))
    return z_t_clean + sigma_t * eps
