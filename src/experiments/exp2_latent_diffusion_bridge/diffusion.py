"""
I²SB latent diffusion bridge — ε-prediction parameterization.
Based on I²SB (Liu et al., 2023): https://arxiv.org/abs/2302.05872

Forward process (I²SB eq. 11, OT/linear case):
  z_t = (1-t)·z_nat + t·z_acc + σ_bridge(t)·ε
  σ_bridge(t) = sigma_max·√(t(1-t))    peaks at t=0.5, zero at endpoints

Training target (I²SB eq. 12):
  label = (z_t - z_nat) / σ_fwd(t)
  σ_fwd(t) = sigma_max·√t              (std of forward process from z_nat)

Recovery (I²SB compute_pred_x0):
  z_nat_hat = z_t - σ_fwd(t)·ε_pred

Inference ODE step (I²SB p_posterior, ot_ode=True):
  z_{t'} = (1 - t'/t)·z_nat_hat + (t'/t)·z_t    uses CURRENT z_t, not z_acc

sigma_max calibration: set to the per-element std of (z_acc - z_nat) in the
target latent space. This ensures the label has ~unit variance at all timesteps.
For Whisper encoder latents: sigma_max ≈ 2.0.

Two alignment modes:
  position — frame-by-frame blend of full 1500-frame sequences (bridge_loss)
  dtw      — DTW-aligned speech + interpolated tail (bridge_loss_dtw)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_forward(z_nat: torch.Tensor, z_acc: torch.Tensor, t: torch.Tensor, sigma_max: float = 0.5) -> torch.Tensor:
    """
    Forward diffusion process: interpolate between z_nat (clean) and z_acc (corrupted).

    z_t = (1-t)·z_nat + t·z_acc + σ_bridge(t)·ε

    where σ_bridge(t) = sigma_max · √(t(1-t)) peaks at t=0.5 and is 0 at both endpoints.

    Args:
        z_nat: [B, L, D] native encoder states (target)
        z_acc: [B, L, D] accented encoder states (corrupted)
        t: [B] timesteps in [0, 1]
        sigma_max: max noise scale

    Returns:
        z_t: [B, L, D] noisy interpolation at timestep t
        eps: [B, L, D] sampled Gaussian noise (kept for debugging)
    """
    B, L, D = z_nat.shape
    device, dtype = z_nat.device, z_nat.dtype

    eps = torch.randn(B, L, D, device=device, dtype=dtype)

    t = t.view(B, 1, 1)

    sigma_t = sigma_max * torch.sqrt((t * (1 - t)).clamp(min=1e-5))
    z_t = (1 - t) * z_nat + t * z_acc + sigma_t * eps

    return z_t, eps


def bridge_loss(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_end: torch.Tensor,
    nat_speech_end: torch.Tensor,
    sigma_max: float = 0.5,
    parameterization: str = "eps",
    pos_tail: str = "fixed",
    tail_weight: float = 0.3,
) -> tuple[torch.Tensor, float, float]:
    """
    Training loss for the position-aligned bridge (I²SB).

    z_nat_canon = z_nat[:T_nat] || z_acc[T_l2:] tiled to fill [T_nat:1500].
    Loss: all-frames MSE against z_nat_canon (no masking).

    pos_tail controls forward process tail construction:
      "fixed" — speech [0:T_nat] linearly interpolated; tail fixed at z_nat_canon
      "full"  — standard I²SB linear blend of all 1500 frames via z_nat_canon

    Args:
        model:            BridgeTransformer
        z_nat:            [B, L, D] native reference states
        z_acc:            [B, L, D] accented input states
        l2_speech_end:    [B] L2 speech end frame — tail source start index
        nat_speech_end:   [B] native speech end frame — z_nat_canon boundary
        sigma_max:        noise schedule parameter
        parameterization: "eps" or "x0"
        pos_tail:         "fixed" or "full"

    Returns:
        loss: scalar MSE over all 1500 frames against z_nat_canon
    """
    B, L, D = z_nat.shape
    device = z_nat.device

    # ── z_nat_canon: z_nat[:T_nat] || z_acc[T_l2:] tiled to [T_nat:L] ────────
    tail_mask    = torch.arange(L, device=device).unsqueeze(0) >= nat_speech_end.unsqueeze(1)   # [B, L]
    tail_offset  = torch.arange(L, device=device).unsqueeze(0) - nat_speech_end.unsqueeze(1)    # [B, L]
    tail_src_idx = (l2_speech_end.unsqueeze(1) + tail_offset).clamp(0, L - 1)                   # [B, L]
    z_acc_tail   = torch.gather(z_acc, 1, tail_src_idx.unsqueeze(-1).expand(-1, -1, D))          # [B, L, D]
    z_nat_canon  = torch.where(tail_mask.unsqueeze(-1), z_acc_tail, z_nat)                       # [B, L, D]

    # ── Forward process ────────────────────────────────────────────────────────
    t = torch.rand(B, device=device)

    if pos_tail == "fixed":
        # Speech [0:T_nat]: interpolate z_nat ↔ z_acc; tail: z_nat_canon (no interp)
        eps      = torch.randn_like(z_acc)
        sigma_t  = (sigma_max * torch.sqrt((t * (1 - t)).clamp(min=1e-5))).view(B, 1, 1)
        t_view   = t.view(B, 1, 1)
        z_t_blend = (1 - t_view) * z_nat + t_view * z_acc
        z_t = torch.where(tail_mask.unsqueeze(-1), z_nat_canon, z_t_blend) + sigma_t * eps
    else:
        # "full": standard I²SB linear blend of all 1500 frames
        z_t, _ = sample_forward(z_nat_canon, z_acc, t, sigma_max=sigma_max)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if parameterization == "x0":
        pred   = model(z_t, t, z_acc)
        target = z_nat_canon
    else:
        t_expanded    = t.view(B, 1, 1)
        sigma_forward = sigma_max * torch.sqrt(t_expanded.clamp(min=1e-5))
        target = (z_t - z_nat_canon) / (sigma_forward + 1e-8)
        pred   = model(z_t, t, z_acc)

    # ── Split loss: speech [0:T_nat] full weight, tail [T_nat:] downweighted ──
    # Use elementwise multiply to avoid expensive non-contiguous gathers
    diff_sq   = (pred - target).pow(2)               # [B, L, D]
    sp_mask_f = (~tail_mask).float().unsqueeze(-1)   # [B, L, 1] — broadcasts over D
    tl_mask_f = tail_mask.float().unsqueeze(-1)
    D = pred.shape[-1]
    speech_loss = (diff_sq * sp_mask_f).sum() / (sp_mask_f.sum() * D)
    tail_loss   = (diff_sq * tl_mask_f).sum() / (tl_mask_f.sum() * D)
    return speech_loss + tail_weight * tail_loss, speech_loss.item(), tail_loss.item()


def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 2.0,
    parameterization: str = "eps",
    speech_end: Optional[int] = None,
) -> torch.Tensor:
    """
    Reverse diffusion: map z_acc (corrupted) → z_nat_hat (corrected).

    Deterministic ODE reverse from t=1→0 (I²SB p_posterior with ot_ode=True).

    At each step t_cur → t_next:
    1. eps_pred   = model(z_t, t_cur)
    2. z_nat_hat  = z_t − σ_fwd(t_cur)·eps_pred        (I²SB compute_pred_x0)
    3. z_t        = (1 − t_next/t_cur)·z_nat_hat
                  + (t_next/t_cur)·z_t                  (I²SB p_posterior OT-ODE)

    Step 3 uses the CURRENT z_t, not the original z_acc, so corrections accumulate
    across steps (at step 1 where t_cur=1 both forms coincide).
    When eps_pred≈0: z_nat_hat = z_t → inference is identity → graceful degradation.

    Args:
        model:     BridgeTransformer for noise prediction
        z_acc:     [B, L, D] accented (corrupted) encoder states
        n_steps:   number of ODE steps
        sigma_max: bridge noise scale — must match training value

    Returns:
        z_nat_hat: [B, L, D] corrected encoder states
    """
    B, L, D = z_acc.shape
    device, dtype = z_acc.device, z_acc.dtype

    t_schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)
    z_t = z_acc.clone()

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t_cur  = t_schedule[i]
            t_next = t_schedule[i + 1]

            t_cur_batch = torch.full((B,), t_cur, device=device, dtype=dtype)

            if parameterization == "x0":
                z_nat_hat = model(z_t, t_cur_batch, z_acc)
            else:
                eps_pred = model(z_t, t_cur_batch, z_acc)
                sigma_forward_cur = sigma_max * torch.sqrt(torch.clamp(t_cur, min=1e-5))
                z_nat_hat = z_t - sigma_forward_cur * eps_pred

            if i == n_steps - 1:
                break

            # I²SB p_posterior (OT-ODE): z_{t'} = (1 - t'/t)·z_nat_hat + (t'/t)·z_t
            z_t = (1 - t_next / t_cur) * z_nat_hat + (t_next / t_cur) * z_t

    if speech_end is not None:
        z_nat_hat[:, speech_end:, :] = z_acc[:, speech_end:, :]
    return z_nat_hat


# ---------------------------------------------------------------------------
# DTW-aligned forward process
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
    T_eng: int,
    T_l2: int,
    eps: np.ndarray,
    sigma_max: float = 0.5,
) -> np.ndarray:
    """DTW-aligned forward process for a single (z_nat, z_acc) pair.

    Speech frames are DTW-interpolated using the alpha-timeline method:
      N(t) = round((1-t)·T_l2 + t·T_eng) output frames
      t_k  = (1-t)·j_norm + t·i_norm      blended path timeline

    Tail frames (N(t)..1499) are linearly interpolated between L2 and native silence.
    Bridge noise σ(t)·ε is added to all 1500 frames.

    path_arr convention (matches precompute_dtw.py):
      path_arr[:, 0] = native frame indices
      path_arr[:, 1] = L2 frame indices

    Args:
        z_nat_np:  [1500, D] native encoder states (float32)
        z_acc_np:  [1500, D] accented encoder states (float32)
        t:         scalar timestep in [0, 1]
        path_arr:  [P, 2] int16 DTW warping path
        T_eng:     native speech end frame
        T_l2:      L2 speech end frame
        eps:       [1500, D] pre-generated Gaussian noise (float32)
        sigma_max: noise schedule parameter

    Returns:
        z_t: [1500, D] noisy DTW-interpolated latent (float32)
    """
    # Bridge convention: t=0=clean=z_nat, t=1=corrupted=z_acc
    # Steering alpha = 1-t, so N and t_k are mirrored from run_steering.py
    N        = max(1, round(t * T_l2 + (1 - t) * T_eng))
    eng_norm = path_arr[:, 0].astype(np.float32) / max(T_eng - 1, 1)
    l2_norm  = path_arr[:, 1].astype(np.float32) / max(T_l2  - 1, 1)

    t_k   = t * l2_norm + (1 - t) * eng_norm
    out_t = np.linspace(0.0, 1.0, N, dtype=np.float32)

    idx_r = np.clip(np.searchsorted(t_k, out_t), 0, len(t_k) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(t_k) - 1)
    k_idx = np.where(
        np.abs(t_k[idx_l] - out_t) <= np.abs(t_k[idx_r] - out_t),
        idx_l, idx_r,
    )
    nat_idx = path_arr[k_idx, 0].astype(np.int32)
    l2_idx  = path_arr[k_idx, 1].astype(np.int32)
    speech  = ((1 - t) * z_nat_np[nat_idx] + t * z_acc_np[l2_idx]).astype(np.float32)

    need = 1500 - N
    if need > 0:
        l2_pad  = _extend_sil(z_acc_np[T_l2:],  need)
        nat_pad = _extend_sil(z_nat_np[T_eng:], need)
        tail    = ((1 - t) * nat_pad + t * l2_pad).astype(np.float32)
        z_clean = np.concatenate([speech, tail], axis=0)
    else:
        z_clean = speech[:1500]

    sigma_t = np.float32(sigma_max * np.sqrt(max(t * (1 - t), 1e-5)))
    return z_clean + sigma_t * eps


def bridge_loss_dtw(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_ends: torch.Tensor,
    nat_speech_ends: torch.Tensor,
    path_tensor: torch.Tensor,
    sigma_max: float = 0.5,
    parameterization: str = "eps",
    dtw_tail: str = "l2",
    tail_weight: float = 0.3,
) -> tuple[torch.Tensor, float, float]:
    """Training loss for DTW-aligned ε-prediction — fully GPU batched.

    Implements the same interpolation logic as sample_forward_dtw but without
    any Python loop or CPU round-trips. Paths are padded to [B, max_P, 2] in
    collate_fn_dtw (last valid path point repeated), which keeps t_k monotone
    so torch.searchsorted stays correct.

    Loss is unmasked ε-prediction MSE over all 1500 frames (I²SB eq. 12),
    matching the position-aligned bridge_loss formulation.

    Args:
        z_nat:          [B, 1500, D] on device
        z_acc:          [B, 1500, D] on device
        l2_speech_ends: [B] on device
        nat_speech_ends:[B] on device
        path_tensor:    [B, max_P, 2] int16 on device — padded DTW paths
                        path_tensor[:, :, 0] = nat indices, [:, :, 1] = l2 indices
    """
    B, _, D   = z_nat.shape
    device    = z_nat.device
    max_P     = path_tensor.shape[1]

    # ── 1. Sample t per item ──────────────────────────────────────────────────
    t_batch = torch.rand(B, device=device, dtype=torch.float32)  # [B]

    # ── 2. Output speech length N(t) = round(t*T_l2 + (1-t)*T_eng) ─────────────
    # Bridge convention: t=0=clean=z_nat (native), t=1=corrupted=z_acc (L2)
    T_l2  = l2_speech_ends.float()   # [B]
    T_eng = nat_speech_ends.float()  # [B]
    N_batch = (t_batch * T_l2 + (1 - t_batch) * T_eng).round().long().clamp(min=1)  # [B]
    max_N   = int(N_batch.max())

    # ── 3. Blended path timeline t_k = t*l2_norm + (1-t)*eng_norm ───────────
    eng_norm = path_tensor[:, :, 0].float() / (T_eng - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    l2_norm  = path_tensor[:, :, 1].float() / (T_l2  - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    t_k      = t_batch[:, None] * l2_norm + (1 - t_batch[:, None]) * eng_norm         # [B, max_P]

    # ── 4. Per-item output grid out_t[b,k] = k / (N_b - 1) ──────────────────
    k_grid = torch.arange(max_N, device=device, dtype=torch.float32).unsqueeze(0)   # [1, max_N]
    out_t  = (k_grid / (N_batch - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)  # [B, max_N]

    # ── 5. Nearest-neighbour lookup on blended timeline ──────────────────────
    idx_r  = torch.searchsorted(t_k.contiguous(), out_t.contiguous()).clamp(0, max_P - 1)  # [B, max_N]
    idx_l  = (idx_r - 1).clamp(0, max_P - 1)
    t_k_r  = torch.gather(t_k, 1, idx_r)
    t_k_l  = torch.gather(t_k, 1, idx_l)
    k_idx  = torch.where((t_k_l - out_t).abs() <= (t_k_r - out_t).abs(), idx_l, idx_r)  # [B, max_N]

    # ── 6. Speech frames: gather DTW-matched nat/l2 frames and blend ─────────
    nat_idx = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx)  # [B, max_N]
    l2_idx  = torch.gather(path_tensor[:, :, 1].long(), 1, k_idx)  # [B, max_N]

    speech_acc = torch.gather(z_acc, 1, l2_idx.unsqueeze(-1).expand(-1, -1, D))   # [B, max_N, D]
    speech_nat = torch.gather(z_nat, 1, nat_idx.unsqueeze(-1).expand(-1, -1, D))  # [B, max_N, D]
    t_view     = t_batch.view(B, 1, 1)
    speech     = (1 - t_view) * speech_nat + t_view * speech_acc                   # [B, max_N, D]

    # ── 7. Tail (silence) frames: alpha-blend L2 and native silence ───────────
    # For item i: silence starts at T_l2_i (L2) / T_eng_i (nat) and runs to 1499.
    # Gather up to max_need = 1500 - min(N) frames; items with larger N use a prefix.
    max_need   = 1500 - int(N_batch.min())
    tail_range = torch.arange(max_need, device=device).unsqueeze(0)                         # [1, max_need]
    tail_i_acc = (T_l2.long().unsqueeze(1)  + tail_range).clamp(0, 1499)                    # [B, max_need]
    tail_i_nat = (T_eng.long().unsqueeze(1) + tail_range).clamp(0, 1499)                    # [B, max_need]
    tail_acc   = torch.gather(z_acc, 1, tail_i_acc.unsqueeze(-1).expand(-1, -1, D))         # [B, max_need, D]
    tail_nat   = torch.gather(z_nat, 1, tail_i_nat.unsqueeze(-1).expand(-1, -1, D))         # [B, max_need, D]
    if dtw_tail == "l2":
        tail = tail_acc
    elif dtw_tail == "english":
        tail = tail_nat
    else:  # interp
        tail = (1 - t_view) * tail_nat + t_view * tail_acc                                  # [B, max_need, D]

    # ── 8. Assemble [B, 1500, D]: speech frames 0..N_i-1, tail N_i..1499 ─────
    pos          = torch.arange(1500, device=device).unsqueeze(0).expand(B, -1)              # [B, 1500]
    speech_mask  = (pos < N_batch.unsqueeze(1))                                               # [B, 1500]

    speech_pos   = pos.clamp(0, max_N - 1)
    speech_out   = torch.gather(speech, 1, speech_pos.unsqueeze(-1).expand(-1, -1, D))       # [B, 1500, D]
    speech_out   = speech_out * speech_mask.unsqueeze(-1)

    tail_pos     = (pos - N_batch.unsqueeze(1)).clamp(0, max_need - 1)
    tail_out     = torch.gather(tail, 1, tail_pos.unsqueeze(-1).expand(-1, -1, D))           # [B, 1500, D]
    tail_out     = tail_out * (~speech_mask).unsqueeze(-1)

    z_clean = speech_out + tail_out                                                            # [B, 1500, D]

    # ── 9. Bridge noise σ_bridge(t)·ε, σ_bridge(t) = sigma_max·√(t(1-t)) ───────
    eps      = torch.randn_like(z_clean)
    sigma_br = (sigma_max * torch.sqrt((t_batch * (1 - t_batch)).clamp(min=1e-5))).view(B, 1, 1)
    z_t      = z_clean + sigma_br * eps

    # ── 10. z_nat_canon: z_nat[:T_nat] || z_acc[T_l2:] tiled to [T_nat:1500] ─
    tail_mask   = nat_speech_ends.unsqueeze(1) <= torch.arange(1500, device=device).unsqueeze(0)  # [B, 1500]
    tail_offset = torch.arange(1500, device=device).unsqueeze(0) - nat_speech_ends.unsqueeze(1)   # [B, 1500]
    tail_src    = (l2_speech_ends.unsqueeze(1) + tail_offset).clamp(0, 1499)                      # [B, 1500]
    z_acc_tail  = torch.gather(z_acc, 1, tail_src.unsqueeze(-1).expand(-1, -1, D))                # [B, 1500, D]
    z_nat_canon = torch.where(tail_mask.unsqueeze(-1), z_acc_tail, z_nat)                         # [B, 1500, D]

    # ── 11. Prediction then split loss: speech [0:T_nat], tail [T_nat:1500] ──
    if parameterization == "x0":
        pred   = model(z_t, t_batch, z_acc)
        target = z_nat_canon
    else:
        sigma_fwd  = (sigma_max * torch.sqrt(t_batch.clamp(min=1e-5))).view(B, 1, 1)
        target = (z_t - z_nat_canon) / (sigma_fwd + 1e-8)
        pred   = model(z_t, t_batch, z_acc)

    diff_sq   = (pred - target).pow(2)
    sp_mask_f = (~tail_mask).float().unsqueeze(-1)
    tl_mask_f = tail_mask.float().unsqueeze(-1)
    D = pred.shape[-1]
    speech_loss = (diff_sq * sp_mask_f).sum() / (sp_mask_f.sum() * D)
    tail_loss   = (diff_sq * tl_mask_f).sum() / (tl_mask_f.sum() * D)
    return speech_loss + tail_weight * tail_loss, speech_loss.item(), tail_loss.item()
