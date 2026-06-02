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

Three parameterizations:
  eps — predicts noise ε; target = (z_t - z_nat) / σ_fwd(t)
  x0  — predicts endpoint z_nat directly (residual on z_t)
  cfm — OT flow matching; predicts velocity v = dz_t/dt; inference is a
         direct Euler integration z_{t-dt} = z_t - v·dt, no x0-blend.
         DTW target: v = z_acc[j_k] - z_nat[i_k] per aligned frame pair.
         Position target: v = z_acc[i] - z_nat[i] per speech frame.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bridge_loss(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_end: torch.Tensor,
    nat_speech_end: torch.Tensor,
    sigma_max: float = 0.5,
    parameterization: str = "eps",
    tail_weight: float = 0.3,
    per_sample: bool = False,
) -> tuple[torch.Tensor, float, float]:
    """
    Training loss for the position-aligned bridge (I²SB).

    N(t) = round(t·T_l2 + (1-t)·T_nat) varies the active speech region each step,
    matching the DTW bridge convention. Speech [0:N(t)] is a positional (frame-by-frame)
    blend; tail [N(t):] copies z_acc[T_l2:] frames — on-manifold silence.
    Bridge noise is added to speech frames only. Inference is shared with bridge_loss_dtw
    via bridge_inference (same N(t) masking, different training blend).

    z_nat_canon = z_nat[:T_nat] || z_acc[T_l2:] placed at [T_nat:].
    Loss: (speech_mask & ~tail_mask) full weight; tail_mask downweighted.
    """
    B, _, D = z_nat.shape
    device  = z_nat.device

    # Crop to active region — same optimisation as bridge_loss_dtw
    max_len = int(max(l2_speech_end.max(), nat_speech_end.max()).item()) + 1
    z_nat   = z_nat[:, :max_len, :]
    z_acc   = z_acc[:, :max_len, :]

    T_l2  = l2_speech_end.float()
    T_nat = nat_speech_end.float()

    # ── 1. Sample t, compute N(t) ─────────────────────────────────────────────
    t_batch = torch.rand(B, device=device, dtype=torch.float32)
    N_batch = (t_batch * T_l2 + (1 - t_batch) * T_nat).round().long().clamp(min=1)  # [B]

    # ── 2. speech_mask: True for active speech frames ─────────────────────────
    pos         = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)  # [B, max_len]
    speech_mask = pos < N_batch.unsqueeze(1)                                        # [B, max_len]

    # ── 3. Forward z_t: positional blend for speech, z_acc[T_l2:] for tail ───
    t_view         = t_batch.view(B, 1, 1)
    z_speech_blend = (1 - t_view) * z_nat + t_view * z_acc                                  # [B, max_len, D]
    tail_offset    = (pos - N_batch.unsqueeze(1)).clamp(min=0)                               # [B, max_len]
    tail_src       = (l2_speech_end.long().unsqueeze(1) + tail_offset).clamp(0, max_len - 1) # [B, max_len]
    z_tail         = torch.gather(z_acc, 1, tail_src.unsqueeze(-1).expand(-1, -1, D))        # [B, max_len, D]
    z_clean        = torch.where(speech_mask.unsqueeze(-1), z_speech_blend, z_tail)

    sigma_br = (sigma_max * torch.sqrt((t_batch * (1 - t_batch)).clamp(min=1e-5))).view(B, 1, 1)
    eps      = torch.randn_like(z_clean)
    z_t      = z_clean + sigma_br * eps * speech_mask.unsqueeze(-1)

    # ── 4. z_nat_canon: z_nat[:T_nat] || z_acc[T_l2:] placed at [T_nat:] ─────
    tail_mask    = nat_speech_end.unsqueeze(1) <= pos                                           # [B, max_len]
    canon_offset = pos - nat_speech_end.unsqueeze(1)
    canon_src    = (l2_speech_end.long().unsqueeze(1) + canon_offset).clamp(0, max_len - 1)    # [B, max_len]
    z_acc_tail   = torch.gather(z_acc, 1, canon_src.unsqueeze(-1).expand(-1, -1, D))
    z_nat_canon  = torch.where(tail_mask.unsqueeze(-1), z_acc_tail, z_nat)                     # [B, max_len, D]

    # ── 5. Prediction ─────────────────────────────────────────────────────────
    if parameterization == "x0":
        pred   = model(z_t, t_batch, z_acc)
        target = z_nat_canon
    elif parameterization == "cfm":
        pred    = model(z_t, t_batch, z_acc)
        # position-aligned velocity: z_acc[i] - z_nat[i] for speech, 0 for tail
        sp_only = (speech_mask & ~tail_mask).unsqueeze(-1)
        target  = (z_acc - z_nat) * sp_only
    else:  # eps
        sigma_fwd = (sigma_max * torch.sqrt(t_batch.clamp(min=1e-5))).view(B, 1, 1)
        target    = (z_t - z_nat_canon) / (sigma_fwd + 1e-8)
        pred      = model(z_t, t_batch, z_acc)

    # ── 6. Split loss: [0:T_nat] full weight, [T_nat:] downweighted ──────────
    diff_sq   = (pred - target).pow(2)
    sp_mask_f = (speech_mask & ~tail_mask).float().unsqueeze(-1)
    tl_mask_f = tail_mask.float().unsqueeze(-1)
    feat_dim  = pred.shape[-1]

    if per_sample:
        # Per-sample losses for min-over-natives val — no backward, no velocity term.
        per_sp = (diff_sq * sp_mask_f).sum(dim=(1, 2)) / (sp_mask_f.sum(dim=(1, 2)).clamp(min=1) * feat_dim)
        per_tl = (diff_sq * tl_mask_f).sum(dim=(1, 2)) / (tl_mask_f.sum(dim=(1, 2)).clamp(min=1) * feat_dim)
        return per_sp + tail_weight * per_tl  # [B]

    speech_loss = (diff_sq * sp_mask_f).sum() / (sp_mask_f.sum() * feat_dim)
    tail_loss   = (diff_sq * tl_mask_f).sum() / (tl_mask_f.sum() * feat_dim)
    return speech_loss + tail_weight * tail_loss, speech_loss.item(), tail_loss.item(), 0.0


def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    T_l2: int,
    T_nat: int,
    n_steps: int = 20,
    sigma_max: float = 2.0,
    parameterization: str = "eps",
) -> torch.Tensor:
    """
    Reverse diffusion with N(t) mask schedule — shared by position and DTW alignments.

    At each ODE step the active speech region is [0:N(t)] where
      N(t) = round(t·T_l2 + (1-t)·T_nat)
    Frames [N(t):] are held at on-manifold silence (z_acc[T_l2:] frames)
    so the bridge never operates on padding frames.

    T_l2 from the mapping JSON; T_nat from TNatPredictor (required for both alignments).

    Args:
        model:     BridgeTransformer
        z_acc:     [B, 1500, 768] accented encoder states
        T_l2:      accented speech end frame
        T_nat:     predicted native speech end frame
        n_steps:   ODE steps
        sigma_max: must match training value
    """
    B, L, D = z_acc.shape
    device, dtype = z_acc.device, z_acc.dtype

    # Crop model inputs to match training distribution — model only saw frames
    # up to max(T_l2, T_nat) during training; frames beyond are OOD in z_acc.
    inf_len = max(T_l2, T_nat) + 1
    z_acc_crop = z_acc[:, :inf_len, :]

    t_schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)
    z_t = z_acc.clone()

    def _sil_src(N_t: int) -> torch.Tensor:
        """Indices into z_acc for the silence tail starting at N_t.
        Mirrors training: tail position k → z_acc[clamp(T_l2 + k, max=L-1)]."""
        return torch.arange(T_l2, T_l2 + (L - N_t), device=device).clamp(max=L - 1)

    def _apply_mask(z: torch.Tensor, t: float) -> torch.Tensor:
        N_t = max(1, round(float(t) * T_l2 + (1 - float(t)) * T_nat))
        N_t = min(N_t, L)
        z[:, N_t:, :] = z_acc[:, _sil_src(N_t), :]
        return z

    # Initialise: at t=1 the tail [T_l2:] is already silence in z_acc, but
    # apply mask to be explicit
    z_t = _apply_mask(z_t, t_schedule[0].item())

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t_cur  = t_schedule[i]
            t_next = t_schedule[i + 1]

            t_cur_batch = torch.full((B,), t_cur, device=device, dtype=dtype)

            if parameterization in ("cfm", "cfm_prewarp"):
                v_pred = model(z_t[:, :inf_len, :], t_cur_batch, z_acc_crop)
                dt = (t_cur - t_next).item()
                z_t[:, :inf_len, :] = z_t[:, :inf_len, :] - v_pred * dt
                if parameterization == "cfm":
                    z_t = _apply_mask(z_t, t_next.item())  # regular CFM morphs timeline
                # cfm_prewarp: no masking — active region stays at T_l2 throughout
                continue

            if parameterization == "x0":
                z_nat_hat_crop = model(z_t[:, :inf_len, :], t_cur_batch, z_acc_crop)
            else:  # eps
                eps_pred = model(z_t[:, :inf_len, :], t_cur_batch, z_acc_crop)
                sigma_forward_cur = sigma_max * torch.sqrt(torch.clamp(t_cur, min=1e-5))
                z_nat_hat_crop = z_t[:, :inf_len, :] - sigma_forward_cur * eps_pred

            # Pad model output back to full L before ODE update and masking
            z_nat_hat = z_t.clone()
            z_nat_hat[:, :inf_len, :] = z_nat_hat_crop

            if i == n_steps - 1:
                break

            z_t = (1 - t_next / t_cur) * z_nat_hat + (t_next / t_cur) * z_t
            if sigma_max > 0:
                # SDE reverse step: inject posterior noise σ(t',t) = sigma_max·√(t'·(t-t')/t).
                # Derived from I²SB p_posterior with σ_fwd(t)=sigma_max·√t.
                # _apply_mask below overwrites the tail with silence, so noise only
                # survives in speech frames [0:N(t')].
                noise_std = sigma_max * torch.sqrt(
                    (t_next * (t_cur - t_next) / t_cur).clamp(min=0.0)
                )
                z_t[:, :inf_len, :] = z_t[:, :inf_len, :] + noise_std * torch.randn(
                    B, inf_len, D, device=device, dtype=dtype
                )
            z_t = _apply_mask(z_t, t_next.item())

    # Enforce silence from T_nat/T_l2 onwards in the final output
    if parameterization == "cfm":
        z_t[:, T_nat:, :] = z_acc[:, _sil_src(T_nat), :]
        return z_t
    if parameterization == "cfm_prewarp":
        z_t[:, T_l2:, :] = z_acc[:, _sil_src(T_l2), :]
        return z_t
    z_nat_hat[:, T_nat:, :] = z_acc[:, _sil_src(T_nat), :]
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
    T_nat: int,
    T_l2: int,
    eps: np.ndarray,
    sigma_max: float = 0.5,
) -> np.ndarray:
    """DTW-aligned forward process for a single (z_nat, z_acc) pair.

    Speech frames are DTW-interpolated using the alpha-timeline method:
      N(t) = round((1-t)·T_l2 + t·T_nat) output frames
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
        T_nat:     native speech end frame
        T_l2:      L2 speech end frame
        eps:       [1500, D] pre-generated Gaussian noise (float32)
        sigma_max: noise schedule parameter

    Returns:
        z_t: [1500, D] noisy DTW-interpolated latent (float32)
    """
    # Bridge convention: t=0=clean=z_nat, t=1=corrupted=z_acc
    # Steering alpha = 1-t, so N and t_k are mirrored from run_steering.py
    N        = max(1, round(t * T_l2 + (1 - t) * T_nat))
    nat_norm = path_arr[:, 0].astype(np.float32) / max(T_nat - 1, 1)
    l2_norm  = path_arr[:, 1].astype(np.float32) / max(T_l2  - 1, 1)

    t_k   = t * l2_norm + (1 - t) * nat_norm
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
        nat_pad = _extend_sil(z_nat_np[T_nat:], need)
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
    lambda_v: float = 0.0,
    per_sample: bool = False,
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

    # Crop to active region — frames beyond max(T_l2, T_nat) in this batch are
    # frozen silence in z_t and trivial offset-mismatch in z_nat_canon.
    # Reduces attention from O(1500²) to O(max_len²); p99 max_len ≈ 380.
    max_len = int(max(l2_speech_ends.max(), nat_speech_ends.max()).item()) + 1
    z_nat   = z_nat[:, :max_len, :]
    z_acc   = z_acc[:, :max_len, :]

    # ── 1. Sample t per item ──────────────────────────────────────────────────
    t_batch = torch.rand(B, device=device, dtype=torch.float32)  # [B]

    # ── 2. Output speech length N(t) = round(t*T_l2 + (1-t)*T_nat) ─────────────
    # Bridge convention: t=0=clean=z_nat (native), t=1=corrupted=z_acc (L2)
    T_l2  = l2_speech_ends.float()   # [B]
    T_nat = nat_speech_ends.float()  # [B]
    N_batch = (t_batch * T_l2 + (1 - t_batch) * T_nat).round().long().clamp(min=1)  # [B]
    max_N   = int(N_batch.max())

    # ── 3. Blended path timeline t_k = t*l2_norm + (1-t)*nat_norm ───────────
    nat_norm = path_tensor[:, :, 0].float() / (T_nat - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    l2_norm  = path_tensor[:, :, 1].float() / (T_l2  - 1).clamp(min=1).unsqueeze(1)  # [B, max_P]
    t_k      = t_batch[:, None] * l2_norm + (1 - t_batch[:, None]) * nat_norm         # [B, max_P]

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
    # For item i: silence starts at T_l2_i (L2) / T_nat_i (nat) and runs to 1499.
    # Gather up to max_need = 1500 - min(N) frames; items with larger N use a prefix.
    max_need   = max_len - int(N_batch.min())
    tail_range = torch.arange(max_need, device=device).unsqueeze(0)                         # [1, max_need]
    tail_i_acc = (T_l2.long().unsqueeze(1)  + tail_range).clamp(0, max_len - 1)             # [B, max_need]
    tail_i_nat = (T_nat.long().unsqueeze(1) + tail_range).clamp(0, max_len - 1)             # [B, max_need]
    tail_acc   = torch.gather(z_acc, 1, tail_i_acc.unsqueeze(-1).expand(-1, -1, D))         # [B, max_need, D]
    tail_nat   = torch.gather(z_nat, 1, tail_i_nat.unsqueeze(-1).expand(-1, -1, D))         # [B, max_need, D]
    if dtw_tail == "l2":
        tail = tail_acc
    elif dtw_tail == "english":
        tail = tail_nat
    else:  # interp
        tail = (1 - t_view) * tail_nat + t_view * tail_acc                                  # [B, max_need, D]

    # ── 8. Assemble [B, max_len, D]: speech frames 0..N_i-1, tail N_i..max_len-1 ─
    pos          = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)            # [B, max_len]
    speech_mask  = (pos < N_batch.unsqueeze(1))                                               # [B, max_len]

    speech_pos   = pos.clamp(0, max_N - 1)
    speech_out   = torch.gather(speech, 1, speech_pos.unsqueeze(-1).expand(-1, -1, D))       # [B, 1500, D]
    speech_out   = speech_out * speech_mask.unsqueeze(-1)

    tail_pos     = (pos - N_batch.unsqueeze(1)).clamp(0, max_need - 1)
    tail_out     = torch.gather(tail, 1, tail_pos.unsqueeze(-1).expand(-1, -1, D))           # [B, 1500, D]
    tail_out     = tail_out * (~speech_mask).unsqueeze(-1)

    z_clean = speech_out + tail_out                                                            # [B, 1500, D]

    # ── 9. Bridge noise σ_bridge(t)·ε — only applied to speech frames [0:N(t)] ──
    eps      = torch.randn_like(z_clean)
    sigma_br = (sigma_max * torch.sqrt((t_batch * (1 - t_batch)).clamp(min=1e-5))).view(B, 1, 1)
    z_t      = z_clean + sigma_br * eps * speech_mask.unsqueeze(-1)

    # ── 10. z_nat_canon: z_nat[:T_nat] || z_acc[T_l2:] tiled to [T_nat:max_len] ─
    tail_mask   = nat_speech_ends.unsqueeze(1) <= torch.arange(max_len, device=device).unsqueeze(0)  # [B, max_len]
    tail_offset = torch.arange(max_len, device=device).unsqueeze(0) - nat_speech_ends.unsqueeze(1)   # [B, max_len]
    tail_src    = (l2_speech_ends.unsqueeze(1) + tail_offset).clamp(0, max_len - 1)                  # [B, max_len]
    z_acc_tail  = torch.gather(z_acc, 1, tail_src.unsqueeze(-1).expand(-1, -1, D))                # [B, 1500, D]
    z_nat_canon = torch.where(tail_mask.unsqueeze(-1), z_acc_tail, z_nat)                         # [B, 1500, D]

    # ── 11. Prediction then split loss ───────────────────────────────────────
    if parameterization == "cfm_prewarp":
        # Prewarp CFM: freeze DTW alignment at t=1, straight-line blend on L2 timeline.
        # At t=1 the blended timeline collapses to l2_norm. Searching l2_norm for
        # k/(T_l2-1) gives nat_idx_pw[k] — the native frame DTW-matched to L2 pos k.
        # This is the fixed warped target; no frame-index morphing during training.
        max_l2    = int(T_l2.max())
        k_grid_pw = torch.arange(max_l2, device=device, dtype=torch.float32).unsqueeze(0)       # [1, max_l2]
        out_t_pw  = (k_grid_pw / (T_l2 - 1).float().clamp(min=1).unsqueeze(1)).clamp(max=1.0)  # [B, max_l2]

        idx_r_pw = torch.searchsorted(l2_norm.contiguous(), out_t_pw.contiguous()).clamp(0, max_P - 1)
        idx_l_pw = (idx_r_pw - 1).clamp(0, max_P - 1)
        k_idx_pw = torch.where(
            (torch.gather(l2_norm, 1, idx_l_pw) - out_t_pw).abs()
            <= (torch.gather(l2_norm, 1, idx_r_pw) - out_t_pw).abs(),
            idx_l_pw, idx_r_pw,
        )
        nat_idx_pw   = torch.gather(path_tensor[:, :, 0].long(), 1, k_idx_pw)           # [B, max_l2]
        z_nat_warped = torch.gather(
            z_nat, 1, nat_idx_pw.clamp(0, max_len - 1).unsqueeze(-1).expand(-1, -1, D),
        )  # [B, max_l2, D]

        # Straight-line forward process: z_t[k] = (1-t)*z_nat_warped[k] + t*z_acc[k]
        z_acc_l2   = z_acc[:, :max_l2, :]
        z_clean_pw = (1 - t_view) * z_nat_warped + t_view * z_acc_l2
        if sigma_max > 0:
            sigma_br_pw = (sigma_max * torch.sqrt((t_batch * (1 - t_batch)).clamp(min=1e-5))).view(B, 1, 1)
            z_clean_pw  = z_clean_pw + sigma_br_pw * torch.randn_like(z_clean_pw)
        z_t_pw               = z_acc.clone()            # tail = z_acc silence
        z_t_pw[:, :max_l2, :] = z_clean_pw

        # Fixed speech mask: pos < T_l2 — no N(t) morphing
        sm_pw = pos < T_l2.long().unsqueeze(1)          # [B, max_len]

        # Velocity target: z_acc[k] - z_nat_warped[k], zero outside L2 speech
        v_tgt                  = torch.zeros_like(z_acc)
        v_tgt[:, :max_l2, :]  = (z_acc_l2 - z_nat_warped) * sm_pw[:, :max_l2].unsqueeze(-1)

        pred     = model(z_t_pw, t_batch, z_acc)
        diff_sq_ = (pred - v_tgt).pow(2)
        sp_f     = sm_pw.float().unsqueeze(-1)
        tl_f     = (~sm_pw).float().unsqueeze(-1)
        Dv       = pred.shape[-1]
        if per_sample:
            per_sp = (diff_sq_ * sp_f).sum(dim=(1, 2)) / (sp_f.sum(dim=(1, 2)).clamp(min=1) * Dv)
            per_tl = (diff_sq_ * tl_f).sum(dim=(1, 2)) / (tl_f.sum(dim=(1, 2)).clamp(min=1) * Dv)
            return per_sp + tail_weight * per_tl
        sp_loss = (diff_sq_ * sp_f).sum() / (sp_f.sum() * Dv)
        tl_loss = (diff_sq_ * tl_f).sum() / (tl_f.sum().clamp(min=1) * Dv)
        return sp_loss + tail_weight * tl_loss, sp_loss.item(), tl_loss.item(), 0.0

    if parameterization == "x0":
        pred   = model(z_t, t_batch, z_acc)
        target = z_nat_canon
    elif parameterization == "cfm":
        pred   = model(z_t, t_batch, z_acc)
        # DTW-aligned velocity: z_acc[j_k] - z_nat[i_k] at each output position,
        # assembled into [B, max_len, D] using the same gather as speech_out (step 8)
        v_raw  = speech_acc - speech_nat                                              # [B, max_N, D]
        v_full = torch.gather(v_raw, 1, speech_pos.unsqueeze(-1).expand(-1, -1, D))  # [B, max_len, D]
        target = v_full * speech_mask.unsqueeze(-1)                                   # tail velocity = 0
    else:  # eps
        sigma_fwd  = (sigma_max * torch.sqrt(t_batch.clamp(min=1e-5))).view(B, 1, 1)
        target = (z_t - z_nat_canon) / (sigma_fwd + 1e-8)
        pred   = model(z_t, t_batch, z_acc)

    diff_sq   = (pred - target).pow(2)
    sp_mask_f = (speech_mask & ~tail_mask).float().unsqueeze(-1)  # [0:min(N(t),T_nat)] only
    tl_mask_f = tail_mask.float().unsqueeze(-1)
    D = pred.shape[-1]

    if per_sample:
        # Per-sample losses for min-over-natives val — no backward, no velocity term.
        per_sp = (diff_sq * sp_mask_f).sum(dim=(1, 2)) / (sp_mask_f.sum(dim=(1, 2)).clamp(min=1) * D)
        per_tl = (diff_sq * tl_mask_f).sum(dim=(1, 2)) / (tl_mask_f.sum(dim=(1, 2)).clamp(min=1) * D)
        return per_sp + tail_weight * per_tl  # [B]

    speech_loss = (diff_sq * sp_mask_f).sum() / (sp_mask_f.sum() * D)
    tail_loss   = (diff_sq * tl_mask_f).sum() / (tl_mask_f.sum() * D)

    # ── 12. Branch B: DTW-direction loss (x0 only, lambda_v > 0) ───────────────
    # Penalises the angle between the model's correction direction and the true
    # correction direction at each DTW-matched frame pair:
    #   dir_pred = x0_pred[nat_frame] - z_acc[l2_frame]
    #   dir_true = z_nat[nat_frame]   - z_acc[l2_frame]
    #   loss = weighted_mean(1 - cos_sim(dir_pred, dir_true))
    # Weighted by ||dir_true|| so frames with large accent deviation dominate
    # and near-identical frames (no correction needed) contribute ~0.
    vel_loss = torch.tensor(0.0, device=device)
    if lambda_v > 0.0:
        if parameterization != "x0":
            raise ValueError("lambda_v > 0 requires parameterization='x0'")
        nat_idx_clip = nat_idx.clamp(0, max_len - 1)
        l2_idx_clip  = l2_idx.clamp(0, max_len - 1)
        exp_idx_nat  = nat_idx_clip.unsqueeze(-1).expand(-1, -1, D)
        exp_idx_l2   = l2_idx_clip.unsqueeze(-1).expand(-1, -1, D)
        x0_at_nat    = torch.gather(pred,  1, exp_idx_nat)   # [B, max_N, D]
        z_nat_at_nat = torch.gather(z_nat, 1, exp_idx_nat)   # [B, max_N, D]
        z_acc_at_l2  = torch.gather(z_acc, 1, exp_idx_l2)    # [B, max_N, D]
        path_speech  = (
            (nat_idx_clip < nat_speech_ends.unsqueeze(1)) &
            (l2_idx_clip  < l2_speech_ends.unsqueeze(1))
        ).float().unsqueeze(-1)                               # [B, max_N, 1]
        dir_pred = x0_at_nat - z_acc_at_l2                   # [B, max_N, D]
        dir_true = z_nat_at_nat - z_acc_at_l2                # [B, max_N, D]
        weight   = dir_true.norm(dim=-1, keepdim=True).clamp(min=1e-4)  # [B, max_N, 1]
        cos      = F.cosine_similarity(dir_pred, dir_true, dim=-1, eps=1e-8).unsqueeze(-1)
        vel_loss = ((1 - cos) * weight * path_speech).sum() \
                   / (weight * path_speech).sum().clamp(min=1e-8)

    total_loss = speech_loss + tail_weight * tail_loss + lambda_v * vel_loss
    return total_loss, speech_loss.item(), tail_loss.item(), vel_loss.item()
