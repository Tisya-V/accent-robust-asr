"""
I²SB (Image-to-Image Schrödinger Bridge) diffusion utilities.
Noise-prediction parameterization for latent space refinement.

Forward process: z_t = (1-t)·z_nat + t·z_acc + σ_bridge(t)·ε
  σ_bridge(t) = sigma_max·√(t(1-t))  — bridge noise (used in sample_forward)

Training target (I²SB eq. 12): (z_t - z_nat) / σ_forward(t)
  σ_forward(t) = sigma_max·√t         — forward variance (∫₀ᵗ β dτ for constant β)

  Expanding: target = √t·(z_acc-z_nat)/sigma_max + √(1-t)·ε
  At t=1: (z_acc-z_nat)/sigma_max  [bounded]
  At t=0: ε                         [bounded]

  Using σ_bridge instead of σ_forward gives √(t/(1-t))·(z_acc-z_nat)/sigma_max + ε,
  which blows up as t→1 — that is the bug the previous code had.

Recovery: z_nat_hat = z_t - σ_forward(t)·ε_pred  (I²SB footnote 1)
Inference: deterministic ODE step z_{t'} = (1-t')·z_nat_hat + t'·z_acc  (Corollary 3.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_forward(z_nat: torch.Tensor, z_acc: torch.Tensor, t: torch.Tensor, sigma_max: float = 0.5) -> torch.Tensor:
    """
    Forward diffusion process: interpolate between z_nat (clean) and z_acc (corrupted).

    z_t = (1-t)·z_nat + t·z_acc + σ(t)·ε

    where σ(t) = sigma_max · √(t(1-t)) is the noise schedule that starts at 0 (t=0)
    and ends at 0 (t=1), peaking at t=0.5.

    Args:
        z_nat: [B, L, D] native encoder states (target)
        z_acc: [B, L, D] accented encoder states (corrupted)
        t: [B] timesteps in [0, 1]
        sigma_max: max noise scale

    Returns:
        z_t: [B, L, D] noisy interpolation at timestep t
        eps: [B, L, D] sampled Gaussian noise (unused by bridge_loss; kept for potential debugging)
    """
    B, L, D = z_nat.shape
    device, dtype = z_nat.device, z_nat.dtype

    # Sample noise once per batch element
    eps = torch.randn(B, L, D, device=device, dtype=dtype)

    # Expand t to [B, 1, 1] for broadcasting
    t = t.view(B, 1, 1)

    # Compute noise scale σ(t) = σ_max · √(t(1-t))
    # At t=0 and t=1: σ=0 (no noise at boundaries)
    # At t=0.5: σ=σ_max/2 (peak noise)
    # Clamp to avoid sqrt(0) NaN in bf16 at boundaries
    sigma_t = sigma_max * torch.sqrt((t * (1 - t)).clamp(min=1e-5))

    # Compute z_t
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
    Training loss for noise prediction.

    1. Sample random t ∈ [0, 1]
    2. Compute noisy z_t via forward process
    3. Predict noise ε_pred from z_t
    4. MSE loss on noise, masked to speech region

    Args:
        model: BridgeTransformer for noise prediction
        z_nat: [B, L, D] native reference states
        z_acc: [B, L, D] accented input states
        speech_end: [B] frame indices where speech ends (for masking padding)
        sigma_max: noise schedule parameter

    Returns:
        loss: scalar loss (mean over speech frames)
    """
    B, L, D = z_nat.shape
    device = z_nat.device

    t = torch.rand(B, device=device)

    # z_t from forward process (raw eps not needed — we derive target from z_t directly)
    z_t, _ = sample_forward(z_nat, z_acc, t, sigma_max=sigma_max)

    # Noise target: (z_t - z_nat) / σ_forward(t)  — see module docstring
    # σ_forward = sigma_max·√t  (forward variance from z_nat side, bounded at t=1)
    t_expanded = t.view(B, 1, 1)
    sigma_forward = sigma_max * torch.sqrt(t_expanded.clamp(min=1e-5))
    eps_target = (z_t - z_nat) / (sigma_forward + 1e-8)

    eps_pred = model(z_t, t)

    if speech_end is not None:
        mask = torch.arange(L, device=device).unsqueeze(0) < speech_end.unsqueeze(1)  # [B, L]
        mask = mask.unsqueeze(-1).expand_as(eps_pred)  # [B, L, D]
        loss = F.mse_loss(eps_pred[mask], eps_target[mask])
    else:
        loss = F.mse_loss(eps_pred, eps_target)

    return loss


def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """
    Reverse diffusion: map z_acc (corrupted) → z_nat_hat (corrected).

    Deterministic ODE reverse from t=1→0 (Corollary 3.5 of I²SB paper).

    At each step:
    1. Predict ε_pred ≈ (z_t - z_nat) / σ_forward(t)
    2. Recover z_nat_hat = z_t - σ_forward(t)·ε_pred  (I²SB footnote 1)
    3. ODE step: z_{t'} = (1-t')·z_nat_hat + t'·z_acc  (no stochastic noise)
    Final step returns z_nat_hat directly.

    Args:
        model: BridgeTransformer for noise prediction
        z_acc: [B, L, D] accented (corrupted) encoder states
        n_steps: number of ODE steps
        sigma_max: noise schedule parameter

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
            eps_pred = model(z_t, t_cur_batch)

            # σ_forward(t) = sigma_max·√t — must match training target denominator
            sigma_forward_cur = sigma_max * torch.sqrt(torch.clamp(t_cur, min=1e-5))

            # Recover z_nat estimate (I²SB footnote 1): X₀ = Xₜ - σ_forward·ε_pred
            z_nat_hat = z_t - sigma_forward_cur * eps_pred

            if i == n_steps - 1:
                break

            # Deterministic ODE step (Corollary 3.5): no stochastic noise injection
            z_t = (1 - t_next) * z_nat_hat + t_next * z_acc

    return z_nat_hat