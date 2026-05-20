"""
I²SB latent diffusion bridge — x₀-prediction parameterization.

Instead of predicting noise ε, the model directly predicts z_nat (the clean native
latent). This is equivalent to ε-prediction up to a reparameterization, but avoids
any division by σ(t), making it numerically cleaner in bf16.

Forward process: z_t = (1-t)·z_nat + t·z_acc + σ_bridge(t)·ε
  σ_bridge(t) = sigma_max·√(t(1-t))

Training objective: MSE(model(z_t, t), z_nat)

Inference (deterministic ODE, Corollary 3.5 of I²SB):
  z_nat_hat = model(z_t, t)          — direct x₀ estimate
  z_{t'} = (1-t')·z_nat_hat + t'·z_acc   — step toward estimate

Two alignment modes:
  position — frame-by-frame blend of full 1500-frame sequences (bridge_loss)
  dtw      — DTW-aligned speech + interpolated tail (bridge_loss_dtw)
"""

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
    speech_end: torch.Tensor,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """
    Training loss for x₀-prediction.

    Model directly predicts z_nat from the noisy interpolation z_t.
    No division by σ(t) — numerically clean at all timesteps.

    Args:
        model: BridgeTransformer — output interpreted as z_nat prediction
        z_nat: [B, L, D] native reference states (target)
        z_acc: [B, L, D] accented input states
        speech_end: [B] frame indices where speech ends (for masking padding)
        sigma_max: noise schedule parameter (only affects z_t sampling)

    Returns:
        loss: scalar MSE over speech frames
    """
    B, L, D = z_nat.shape
    device = z_nat.device

    t = torch.rand(B, device=device)
    z_t, _ = sample_forward(z_nat, z_acc, t, sigma_max=sigma_max)

    z_nat_pred = model(z_t, t)

    if speech_end is not None:
        mask = torch.arange(L, device=device).unsqueeze(0) < speech_end.unsqueeze(1)  # [B, L]
        mask = mask.unsqueeze(-1).expand_as(z_nat_pred)  # [B, L, D]
        loss = F.mse_loss(z_nat_pred[mask], z_nat[mask])
    else:
        loss = F.mse_loss(z_nat_pred, z_nat)

    return loss


def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """
    Reverse diffusion: map z_acc (corrupted) → z_nat_hat (corrected).

    Deterministic ODE reverse from t=1→0 (Corollary 3.5 of I²SB).

    At each step:
    1. z_nat_hat = model(z_t, t)              — direct x₀ prediction
    2. z_{t'} = (1-t')·z_nat_hat + t'·z_acc  — ODE step (no noise injection)

    Args:
        model: BridgeTransformer — output interpreted as z_nat prediction
        z_acc: [B, L, D] accented (corrupted) encoder states
        n_steps: number of ODE steps
        sigma_max: unused at inference (kept for API consistency)

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
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]

            t_cur_batch = torch.full((B,), t_cur, device=device, dtype=dtype)
            z_nat_hat = model(z_t, t_cur_batch)

            if i == n_steps - 1:
                break

            # Deterministic ODE step: interpolate toward z_nat_hat estimate
            z_t = (1 - t_next) * z_nat_hat + t_next * z_acc

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
        sigma_max: noise schedule parameter

    Returns:
        z_t: [1500, D] noisy DTW-interpolated latent (float32)
    """
    D = z_nat_np.shape[1]

    N     = max(1, round((1 - t) * T_l2 + t * T_eng))
    i_norm = path_arr[:, 0].astype(np.float32) / max(T_eng - 1, 1)
    j_norm = path_arr[:, 1].astype(np.float32) / max(T_l2  - 1, 1)

    t_k   = (1 - t) * j_norm + t * i_norm
    out_t = np.linspace(0.0, 1.0, N, dtype=np.float32)

    idx_r = np.clip(np.searchsorted(t_k, out_t), 0, len(t_k) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(t_k) - 1)
    k_idx = np.where(
        np.abs(t_k[idx_l] - out_t) <= np.abs(t_k[idx_r] - out_t),
        idx_l, idx_r,
    )
    nat_idx = path_arr[k_idx, 0].astype(np.int32)
    l2_idx  = path_arr[k_idx, 1].astype(np.int32)
    speech  = ((1 - t) * z_acc_np[l2_idx] + t * z_nat_np[nat_idx]).astype(np.float32)

    need = 1500 - N
    if need > 0:
        l2_pad  = _extend_sil(z_acc_np[T_l2:],  need)
        nat_pad = _extend_sil(z_nat_np[T_eng:], need)
        tail    = ((1 - t) * l2_pad + t * nat_pad).astype(np.float32)
        z_clean = np.concatenate([speech, tail], axis=0)
    else:
        z_clean = speech[:1500]

    sigma_t = sigma_max * np.sqrt(max(t * (1 - t), 1e-5))
    eps     = np.random.randn(1500, D).astype(np.float32)
    return z_clean + sigma_t * eps


def bridge_loss_dtw(
    model: nn.Module,
    z_nat: torch.Tensor,
    z_acc: torch.Tensor,
    l2_speech_ends: torch.Tensor,
    nat_speech_ends: torch.Tensor,
    paths: list,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """Training loss for DTW-aligned x₀-prediction.

    Constructs z_t via sample_forward_dtw (numpy, per-item loop since paths have
    variable length), then passes the full [B, 1500, D] batch to the model.
    Loss is unmasked MSE over all 1500 frames — includes silence correction.

    Args:
        model:           BridgeTransformer
        z_nat:           [B, 1500, D] native encoder states
        z_acc:           [B, 1500, D] accented encoder states
        l2_speech_ends:  [B] L2 speech end frames
        nat_speech_ends: [B] native speech end frames
        paths:           list[np.ndarray] of shape [P_i, 2] per item
        sigma_max:       noise schedule parameter
    """
    B, L, D   = z_nat.shape
    device    = z_nat.device
    dtype     = z_nat.dtype

    t_vals      = np.random.rand(B).astype(np.float32)
    z_nat_np    = z_nat.float().cpu().numpy()
    z_acc_np    = z_acc.float().cpu().numpy()
    l2_ends_np  = l2_speech_ends.cpu().numpy()
    nat_ends_np = nat_speech_ends.cpu().numpy()

    z_t_np = np.stack([
        sample_forward_dtw(
            z_nat_np[i], z_acc_np[i],
            float(t_vals[i]),
            paths[i],
            int(nat_ends_np[i]),
            int(l2_ends_np[i]),
            sigma_max=sigma_max,
        )
        for i in range(B)
    ])

    z_t      = torch.from_numpy(z_t_np).to(device=device, dtype=dtype)
    t_tensor = torch.from_numpy(t_vals).to(device=device, dtype=dtype)

    z_nat_pred = model(z_t, t_tensor)
    return F.mse_loss(z_nat_pred, z_nat)
