"""
I²SB (Image-to-Image Schrödinger Bridge) diffusion utilities.
Noise-prediction parameterization for latent space refinement.
"""

import torch
import torch.nn as nn


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
        eps: [B, L, D] sampled Gaussian noise (for loss computation)
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
    sigma_t = sigma_max * torch.sqrt(t * (1 - t))

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
        loss: scalar loss (mean over batch and time)
    """
    B, L, D = z_nat.shape
    device = z_nat.device

    # Sample random timesteps for each batch element
    t = torch.rand(B, device=device)

    # Compute noisy z_t and sample noise
    z_t, eps = sample_forward(z_nat, z_acc, t, sigma_max=sigma_max)

    # Predict noise
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eps_pred = model(z_t, t)

    # MSE on noise prediction
    loss = torch.mean((eps_pred - eps) ** 2)

    # Optional: mask padding frames (frames after speech_end)
    if speech_end is not None:
        mask = torch.arange(L, device=device).unsqueeze(0) < speech_end.unsqueeze(1)  # [B, L]
        mask = mask.float().unsqueeze(-1)  # [B, L, 1]
        loss = torch.sum((eps_pred - eps) ** 2 * mask) / (torch.sum(mask) + 1e-8)

    return loss


def bridge_inference(
    model: nn.Module,
    z_acc: torch.Tensor,
    n_steps: int = 20,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """
    Reverse diffusion: map z_acc (corrupted) → z_nat_hat (corrected).

    Uses deterministic ODE reverse from t=1→0.

    Intuition:
    - At t=0: z_t ≈ z_nat (clean, target)
    - At t=1: z_t ≈ z_acc (corrupted, input)
    - We reverse: start at z_acc, iteratively predict noise and step toward z_nat

    At each step, we:
    1. Predict noise ε_pred from z_t
    2. Recover implicit z_nat_hat = (z_t - t·z_acc - σ(t)·ε_pred) / (1-t)
    3. Interpolate toward it: z_{t-dt} = (1-(t-dt))·z_nat_hat + (t-dt)·z_acc + σ(t-dt)·ε_pred

    Args:
        model: BridgeTransformer for noise prediction
        z_acc: [B, L, D] accented (corrupted) encoder states
        n_steps: number of ODE steps (trade-off: more steps → better quality but slower)
        sigma_max: noise schedule parameter

    Returns:
        z_nat_hat: [B, L, D] corrected (denoised) encoder states
    """
    B, L, D = z_acc.shape
    device, dtype = z_acc.device, z_acc.dtype

    # Time schedule: reverse from t=1 to t=0 in n_steps
    # Using uniform steps for simplicity (could use more sophisticated schedules)
    t_schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)

    # Start with z_t ≈ z_acc (at t=1)
    z_t = z_acc.clone()

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_cur  # Negative (going backward in time)

            # Current timestep tensor [B]
            t_cur_batch = torch.full((B,), t_cur, device=device, dtype=dtype)

            # Predict noise at current step
            eps_pred = model(z_t, t_cur_batch)

            # Compute noise scales
            sigma_t_cur = sigma_max * torch.sqrt(t_cur * (1 - t_cur))
            sigma_t_next = sigma_max * torch.sqrt(t_next * (1 - t_next))

            # Implicit z_nat estimation:
            # From z_t = (1-t)·z_nat + t·z_acc + σ(t)·ε_pred
            # Solve for z_nat: z_nat = (z_t - t·z_acc - σ(t)·ε_pred) / (1-t)
            z_nat_hat = (z_t - t_cur * z_acc - sigma_t_cur.view(-1, 1, 1) * eps_pred) / (1 - t_cur + 1e-8)

            # Step toward next timestep using implicit z_nat
            z_t = (1 - t_next) * z_nat_hat + t_next * z_acc + sigma_t_next.view(-1, 1, 1) * eps_pred

    return z_t