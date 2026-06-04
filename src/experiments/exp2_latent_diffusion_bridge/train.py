#!/usr/bin/env python3
"""
Training loop for E2 Latent Diffusion Bridge.
Plain PyTorch with bf16 AMP, gradient clipping, and checkpoint resumption.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import BridgeTransformer
from .diffusion import bridge_loss, bridge_loss_dtw
from .dataset import BridgeDataset


def collate_fn(batch):
    """Stack batch into tensors (position alignment). Returns slot_ids as list of strings."""
    z_accs, z_nats, l2_ends, nat_ends, slot_ids = zip(*batch)
    return (
        torch.stack(z_accs),        # [B, 1500, 768]
        torch.stack(z_nats),        # [B, 1500, 768]
        torch.tensor(l2_ends),      # [B]
        torch.tensor(nat_ends),     # [B]
        list(slot_ids),             # [B] list of str — l2_encoder_state_path
    )


def collate_fn_dtw(batch):
    """Stack tensors and pad DTW paths to [B, max_P, 2] for batched GPU ops. Returns slot_ids."""
    z_accs, z_nats, l2_ends, nat_ends, paths, slot_ids = zip(*batch)
    max_P  = max(len(p) for p in paths)
    padded = np.zeros((len(paths), max_P, 2), dtype=np.int16)
    for i, p in enumerate(paths):
        padded[i, :len(p)] = p
        padded[i, len(p):] = p[-1]  # repeat last path point — keeps t_k monotone for searchsorted
    return (
        torch.stack(z_accs),            # [B, 1500, 768]
        torch.stack(z_nats),            # [B, 1500, 768]
        torch.tensor(l2_ends),          # [B]
        torch.tensor(nat_ends),         # [B]
        torch.from_numpy(padded),       # [B, max_P, 2] int16
        list(slot_ids),                 # [B] list of str — l2_encoder_state_path
    )


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
):
    """Save checkpoint with model, optimizer, and scheduler state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved: {ckpt_path}")


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> Tuple[int, float]:
    """Load checkpoint and return starting epoch + best val loss."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Handle torch.compile wrapper: strip _orig_mod. prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    print(f"[Checkpoint] Resumed from epoch {epoch}, best_val_loss={best_val_loss:.6f}")
    return epoch, best_val_loss


def save_config(config_path: Path, **kwargs):
    """Save training config to JSON."""
    config = {k: v for k, v in kwargs.items() if not callable(v) and not isinstance(v, (Path, type))}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"[Config] Saved: {config_path}")


def save_history(history_path: Path, train_losses: list, val_losses: list,
                 train_speech_losses: list = None, train_tail_losses: list = None,
                 val_speech_losses: list = None, val_tail_losses: list = None,
                 cosine_sims: dict = None):
    """Save training history to JSON."""
    history = {"train_losses": train_losses, "val_losses": val_losses}
    if train_speech_losses: history["train_speech_losses"] = train_speech_losses
    if train_tail_losses:   history["train_tail_losses"]   = train_tail_losses
    if val_speech_losses:   history["val_speech_losses"]   = val_speech_losses
    if val_tail_losses:     history["val_tail_losses"]     = val_tail_losses
    if cosine_sims:         history["cosine_similarities"] = cosine_sims
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[History] Saved: {history_path}")


def plot_losses(plot_path: Path, train_losses: list, val_losses: list):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {plot_path}")
    plt.close()


_DTW_TAIL_MAP = {"dtw": "l2", "dtw_l2pad": "l2", "dtw_engpad": "english", "dtw_fixed": "l2"}


def _compute_loss(model, batch, device, sigma_max, alignment, parameterization="eps",
                  tail_weight=0.3, lambda_v=0.0):
    """Dispatch to position or DTW loss. Returns (total_loss, speech_loss, tail_loss, vel_loss)."""
    if alignment.startswith("dtw"):
        z_acc, z_nat, l2_speech_end, nat_speech_end, path_tensor, _slot_ids = batch
        z_acc          = z_acc.to(device, non_blocking=True)
        z_nat          = z_nat.to(device, non_blocking=True)
        l2_speech_end  = l2_speech_end.to(device, non_blocking=True)
        nat_speech_end = nat_speech_end.to(device, non_blocking=True)
        path_tensor    = path_tensor.to(device, non_blocking=True)
        dtw_tail       = _DTW_TAIL_MAP.get(alignment, "l2")
        return bridge_loss_dtw(model, z_nat, z_acc, l2_speech_end, nat_speech_end,
                               path_tensor, sigma_max=sigma_max,
                               parameterization=parameterization, alignment=alignment,
                               dtw_tail=dtw_tail, tail_weight=tail_weight, lambda_v=lambda_v)
    else:
        z_acc, z_nat, l2_speech_end, nat_speech_end, _slot_ids = batch
        z_acc          = z_acc.to(device, non_blocking=True)
        z_nat          = z_nat.to(device, non_blocking=True)
        l2_speech_end  = l2_speech_end.to(device, non_blocking=True)
        nat_speech_end = nat_speech_end.to(device, non_blocking=True)
        return bridge_loss(model, z_nat, z_acc, l2_speech_end, nat_speech_end,
                           sigma_max=sigma_max, parameterization=parameterization,
                           tail_weight=tail_weight)


def _compute_loss_per_sample(model, batch, device, sigma_max, alignment,
                              parameterization="eps", tail_weight=0.3) -> torch.Tensor:
    """Return per-sample losses [B] for min-over-natives val computation."""
    if alignment.startswith("dtw"):
        z_acc, z_nat, l2_speech_end, nat_speech_end, path_tensor, _slot_ids = batch
        z_acc          = z_acc.to(device, non_blocking=True)
        z_nat          = z_nat.to(device, non_blocking=True)
        l2_speech_end  = l2_speech_end.to(device, non_blocking=True)
        nat_speech_end = nat_speech_end.to(device, non_blocking=True)
        path_tensor    = path_tensor.to(device, non_blocking=True)
        dtw_tail       = _DTW_TAIL_MAP.get(alignment, "l2")
        return bridge_loss_dtw(model, z_nat, z_acc, l2_speech_end, nat_speech_end,
                               path_tensor, sigma_max=sigma_max,
                               parameterization=parameterization, alignment=alignment,
                               dtw_tail=dtw_tail, tail_weight=tail_weight, per_sample=True)[0]
    else:
        z_acc, z_nat, l2_speech_end, nat_speech_end, _slot_ids = batch
        z_acc          = z_acc.to(device, non_blocking=True)
        z_nat          = z_nat.to(device, non_blocking=True)
        l2_speech_end  = l2_speech_end.to(device, non_blocking=True)
        nat_speech_end = nat_speech_end.to(device, non_blocking=True)
        return bridge_loss(model, z_nat, z_acc, l2_speech_end, nat_speech_end,
                           sigma_max=sigma_max, parameterization=parameterization,
                           tail_weight=tail_weight, per_sample=True)[0]


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma_max: float = 0.5,
    alignment: str = "position",
    parameterization: str = "eps",
    tail_weight: float = 0.3,
    lambda_v: float = 0.0,
) -> tuple[float, float, float, float]:
    """Train for one epoch. Returns (avg_total, avg_speech, avg_tail, avg_vel)."""
    model.train()
    total_loss = speech_loss_sum = tail_loss_sum = vel_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, sp_loss, tl_loss, vel_loss = _compute_loss(
                model, batch, device, sigma_max, alignment, parameterization, tail_weight, lambda_v)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        speech_loss_sum += sp_loss
        tail_loss_sum   += tl_loss
        vel_loss_sum    += vel_loss
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}", "sp": f"{sp_loss:.4f}"}
        if tail_weight > 0:
            postfix["tl"] = f"{tl_loss:.4f}"
        if lambda_v > 0:
            postfix["vl"] = f"{vel_loss:.4f}"
        pbar.set_postfix(postfix)

    pbar.close()
    return total_loss / num_batches, speech_loss_sum / num_batches, tail_loss_sum / num_batches, vel_loss_sum / num_batches


def train_epoch_profile(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma_max: float = 0.5,
) -> float:
    """Profile-instrumented training epoch with detailed timing. Returns average loss."""
    import time

    model.train()
    total_loss = 0.0
    num_batches = 0

    # Timing buckets
    time_load = 0.0
    time_fwd = 0.0
    time_bwd = 0.0
    time_opt = 0.0
    time_todevice = 0.0
    time_zgrad = 0.0

    t_batch_start = time.time()
    iterator = iter(train_loader)
    pbar = tqdm(total=len(train_loader), desc="Training [PROFILE]")

    while num_batches < len(train_loader):
        # Time data loading
        t_load_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        z_acc, z_nat, l2_speech_end = batch[0], batch[1], batch[2]
        time_load += time.time() - t_load_start

        t_todevice_start = time.time()
        z_acc         = z_acc.to(device, non_blocking=True)
        z_nat         = z_nat.to(device, non_blocking=True)
        l2_speech_end = l2_speech_end.to(device, non_blocking=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time_todevice += time.time() - t_todevice_start

        t_zgrad_start = time.time()
        optimizer.zero_grad()
        torch.cuda.synchronize() if device.type == "cuda" else None
        time_zgrad += time.time() - t_zgrad_start

        # Forward pass with bf16 autocast
        t_fwd_start = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = bridge_loss(model, z_nat, z_acc, None, sigma_max=sigma_max)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time_fwd += time.time() - t_fwd_start

        # Backward
        t_bwd_start = time.time()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time_bwd += time.time() - t_bwd_start

        # Optimizer step
        t_opt_start = time.time()
        optimizer.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        time_opt += time.time() - t_opt_start

        total_loss += loss.item()
        num_batches += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if num_batches >= 20:
            break

    pbar.close()

    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = time.time() - t_batch_start
    print(f"\n[Profile - first {num_batches} batches]")
    print(f"  Total:      {total_time:.2f}s ({num_batches/total_time:.2f} it/s)")
    print(f"  Load:       {time_load:.2f}s ({100*time_load/total_time:.1f}%) [disk I/O + tensor creation]")
    print(f"  ToDevice:   {time_todevice:.2f}s ({100*time_todevice/total_time:.1f}%) [host→GPU transfer]")
    print(f"  ZeroGrad:   {time_zgrad:.2f}s ({100*time_zgrad/total_time:.1f}%) [optimizer.zero_grad()]")
    print(f"  Forward:    {time_fwd:.2f}s ({100*time_fwd/total_time:.1f}%)")
    print(f"  Backward:   {time_bwd:.2f}s ({100*time_bwd/total_time:.1f}%)")
    print(f"  Optimizer:  {time_opt:.2f}s ({100*time_opt/total_time:.1f}%)")
    total_accounted = time_load + time_todevice + time_zgrad + time_fwd + time_bwd + time_opt
    print(f"  Unaccounted: {total_time - total_accounted:.2f}s ({100*(total_time - total_accounted)/total_time:.1f}%)")
    # GPU memory stats
    if device.type == "cuda":
        print(f"\n[GPU Memory]")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    avg_loss = total_loss / num_batches
    return avg_loss


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    sigma_max: float = 0.5,
    alignment: str = "position",
    parameterization: str = "eps",
    tail_weight: float = 0.3,
    lambda_v: float = 0.0,  # unused in val; kept for call-site consistency
) -> tuple[float, float, float]:
    """Validate with min-over-natives loss.

    For each L2 utterance slot (identified by l2_encoder_state_path), takes the
    minimum loss over all native speaker pairings present in the val set. This
    measures 'can the model map to any valid native realization?' rather than
    penalising a correct CLB output when the val entry happens to be SLT.

    Returns (avg_min_slot_loss, 0.0, 0.0). Speech/tail breakdown not available
    per-sample; the 0.0 placeholders preserve the return type for save_history.
    """
    model.eval()
    slot_losses: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            slot_ids = batch[-1]  # list of str, last element from collate_fn / collate_fn_dtw
            with torch.autocast("cuda", dtype=torch.bfloat16):
                per_sample = _compute_loss_per_sample(
                    model, batch, device, sigma_max, alignment, parameterization, tail_weight)
            for loss_val, sid in zip(per_sample.tolist(), slot_ids):
                slot_losses[sid].append(loss_val)
            pbar.set_postfix({"slots": len(slot_losses), "cur_min": f"{per_sample.min().item():.4f}"})

    slot_min  = [min(v) for v in slot_losses.values()]
    val_loss  = sum(slot_min) / max(len(slot_min), 1)
    return val_loss, 0.0, 0.0


def train(
    mapping_train_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
    mapping_val_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
    out_dir: str = "models/bridge",
    alignment: str = "position",
    dtw_cache: str = "src/experiments/exp2_latent_diffusion_bridge/dtw_cache/dtw_paths.pkl",
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    sigma_max: float = 2.0,
    cond_acc: bool = False,
    parameterization: str = "eps",
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    dim_feedforward: int = 1024,
    num_workers: int = 4,
    profile: bool = False,
    notes: str = "",
    patience: int = 5,
    seed: int = 42,
    tail_weight: float = 0.3,
    lambda_v: float = 0.0,
):
    """
    Train the bridge model.

    Args:
        mapping_train_path: path to training mapping JSON
        mapping_val_path: path to validation mapping JSON
        out_dir: checkpoint output directory
        n_epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: AdamW weight decay
        sigma_max: diffusion noise schedule parameter
        num_workers: DataLoader workers
        notes: free-text note saved to config.json (e.g. experiment description)
        patience: early stopping patience (epochs without val loss improvement)
        seed: random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Train] Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"[Train] Loading training data from {mapping_train_path} (alignment={alignment})")
    train_dataset = BridgeDataset(mapping_train_path, split="train", alignment=alignment,
                                  dtw_cache=dtw_cache)

    print(f"[Train] Loading validation data from {mapping_val_path}")
    val_dataset = BridgeDataset(mapping_val_path, split="dev", alignment=alignment,
                                dtw_cache=dtw_cache)

    print(f"[Train] Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")

    def _worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    _collate = collate_fn_dtw if alignment.startswith("dtw") else collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )

    # Model
    print(f"[Train] Initializing BridgeTransformer")
    model = BridgeTransformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                              dim_feedforward=dim_feedforward,
                              cond_acc=cond_acc, parameterization=parameterization)
    model = model.to(device=device, dtype=torch.bfloat16)
    

    print(f"[Train] Model: {model}")

    # Compile for speedup (disabled due to multiprocessing conflicts with torch.compile + DataLoader)
    # print(f"[Train] Compiling model...")
    # model = torch.compile(model)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Resume from checkpoint if it exists
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if ckpt_latest.exists():
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch

    print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
    print(sum(p.numel() for p in model.parameters()), "params")

    # Save config
    config = {
        "alignment": alignment,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "sigma_max": sigma_max,
        "cond_acc": cond_acc,
        "parameterization": parameterization,
        "num_workers": num_workers,
        "model": "BridgeTransformer",
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dim_feedforward": dim_feedforward,
        "patience": patience,
        "seed": seed,
        "notes": notes,
        "tail_weight": tail_weight,
        "lambda_v": lambda_v,
    }
    save_config(out_dir / "config.json", **config)

    if lambda_v > 0.0 and not alignment.startswith("dtw"):
        raise ValueError("lambda_v > 0 requires --alignment dtw (velocity loss is DTW-only)")
    if parameterization == "cfm_prewarp" and not alignment.startswith("dtw"):
        raise ValueError("--parameterization cfm_prewarp requires --alignment dtw")

    # Capture a sample batch for sanity checks (cosine similarity of denoised latents)
    # Use generic indexing — batch is 4-tuple (position) or 5-tuple (dtw)
    sample_batch = next(iter(train_loader))
    z_acc_sample          = sample_batch[0][:32].to(device, non_blocking=True)
    z_nat_sample          = sample_batch[1][:32].to(device, non_blocking=True)
    l2_speech_end_sample  = sample_batch[2][:32]   # keep on CPU — used as Python ints
    nat_speech_end_sample = sample_batch[3][:32].to(device, non_blocking=True)
    # DTW paths for cfm_prewarp: needed to compute z_nat_warped diagnostic
    path_sample = sample_batch[4][:32].to(device, non_blocking=True) if alignment.startswith("dtw") else None

    # For cfm_prewarp: precompute z_nat_warped (DTW-aligned native frames on L2 timeline)
    # used as the primary reference for cosine sim (not position-aligned z_nat).
    z_nat_warped_sample = None
    if parameterization == "cfm_prewarp" and path_sample is not None:
        with torch.no_grad():
            B_s   = z_nat_sample.shape[0]
            max_P = path_sample.shape[1]
            T_l2s = l2_speech_end_sample.float().to(device)
            T_nts = nat_speech_end_sample.float()
            l2_n  = path_sample[:, :, 1].float() / (T_l2s - 1).clamp(min=1).unsqueeze(1)
            max_l2s = int(T_l2s.max())
            kg    = torch.arange(max_l2s, device=device, dtype=torch.float32).unsqueeze(0)
            ot    = (kg / (T_l2s - 1).clamp(min=1).unsqueeze(1)).clamp(max=1.0)
            ir    = torch.searchsorted(l2_n.contiguous(), ot.contiguous()).clamp(0, max_P - 1)
            il    = (ir - 1).clamp(0, max_P - 1)
            ki    = torch.where((torch.gather(l2_n, 1, il) - ot).abs()
                                <= (torch.gather(l2_n, 1, ir) - ot).abs(), il, ir)
            ni    = torch.gather(path_sample[:, :, 0].long(), 1, ki)
            D_s   = z_nat_sample.shape[-1]
            max_l_s = z_nat_sample.shape[1]
            z_nat_warped_sample = torch.gather(
                z_nat_sample, 1, ni.clamp(0, max_l_s - 1).unsqueeze(-1).expand(-1, -1, D_s)
            )  # [B, max_l2, D]

    # Baseline cosine similarity before any bridge correction.
    # cfm_prewarp: mask to T_l2 (L2 timeline) and also show cos vs z_nat_warped target.
    # Other parameterizations: mask to T_nat, cos vs z_nat.
    baseline_sim = None
    if z_acc_sample is not None:
        if parameterization == "cfm_prewarp":
            mask = (torch.arange(z_acc_sample.shape[1], device=device).unsqueeze(0)
                    < l2_speech_end_sample.to(device).unsqueeze(1))
            sim_pos = (F.cosine_similarity(z_acc_sample, z_nat_sample, dim=-1) * mask).sum(1) / mask.sum(1).float()
            print(f"[Train] Baseline cos(z_acc, z_nat) on L2 frames:        {sim_pos.mean().item():.4f}")
            if z_nat_warped_sample is not None:
                mw    = mask[:, :z_nat_warped_sample.shape[1]]
                sw    = (F.cosine_similarity(z_acc_sample[:, :z_nat_warped_sample.shape[1]],
                                             z_nat_warped_sample, dim=-1) * mw).sum(1) / mw.sum(1).float()
                baseline_sim = sw.mean().item()  # primary baseline: gap to actual training target
                print(f"[Train] Baseline cos(z_acc, z_nat_warped) on L2 frames: {baseline_sim:.4f}")
            else:
                baseline_sim = sim_pos.mean().item()
        else:
            mask = (torch.arange(z_acc_sample.shape[1], device=device).unsqueeze(0)
                    < nat_speech_end_sample.unsqueeze(1))
            sim_per = (F.cosine_similarity(z_acc_sample, z_nat_sample, dim=-1) * mask).sum(1) / mask.sum(1).float()
            baseline_sim = sim_per.mean().item()
            print(f"[Train] Baseline cosine sim (z_acc vs z_nat, speech frames only): {baseline_sim:.4f}")

    # Training loop with loss history
    train_losses = []; train_speech_losses = []; train_tail_losses = []
    val_losses   = []; val_speech_losses   = []; val_tail_losses   = []
    cosine_sims = {}
    epochs_no_improve = 0
    print(f"[Train] Starting training for {n_epochs} epochs (patience={patience})")
    for epoch in range(start_epoch, n_epochs):
        print(f"\n[Epoch {epoch+1}/{n_epochs}]")

        # Train
        if profile:
            train_loss = train_epoch_profile(model, train_loader, optimizer, device, sigma_max=sigma_max)
            break  # Profile only runs once
        else:
            train_loss, tr_sp, tr_tl, tr_vl = train_epoch(
                model, train_loader, optimizer, device,
                sigma_max=sigma_max, alignment=alignment,
                parameterization=parameterization, tail_weight=tail_weight,
                lambda_v=lambda_v)
        parts = [f"speech={tr_sp:.6f}"]
        if tail_weight > 0:
            parts.append(f"tail={tr_tl:.6f}")
        if lambda_v > 0:
            parts.append(f"dir={tr_vl:.6f}")
        print(f"  Train loss: {train_loss:.6f}  ({',  '.join(parts)})")
        train_losses.append(train_loss); train_speech_losses.append(tr_sp); train_tail_losses.append(tr_tl)

        # Validate (min-over-natives — speech/tail breakdown not available)
        val_loss, val_sp, val_tl = val_epoch(
            model, val_loader, device, sigma_max=sigma_max, alignment=alignment,
            parameterization=parameterization, tail_weight=tail_weight, lambda_v=lambda_v)
        print(f"  Val loss (min-over-natives): {val_loss:.6f}")
        val_losses.append(val_loss); val_speech_losses.append(val_sp); val_tail_losses.append(val_tl)

        # Sanity check: cosine sim + prediction scale on fixed sample mini-batch
        if z_acc_sample is not None:
            from .diffusion import bridge_inference
            with torch.no_grad():
                z_hats = []
                for b in range(z_acc_sample.shape[0]):
                    z_hats.append(bridge_inference(
                        model, z_acc_sample[b:b+1],
                        T_l2=int(l2_speech_end_sample[b].item()),
                        T_nat=int(nat_speech_end_sample[b].item()),
                        n_steps=20, sigma_max=sigma_max,
                        parameterization=parameterization,
                    ))
                z_hat = torch.cat(z_hats, dim=0)
                L = z_hat.shape[1]

                # cfm_prewarp: output lives on L2 timeline — mask and reference differ
                if parameterization == "cfm_prewarp":
                    sp_mask = (torch.arange(L, device=device).unsqueeze(0)
                               < l2_speech_end_sample.to(device).unsqueeze(1))  # [B, L]
                    # Primary: cos(z_hat, z_nat_warped) — the actual training target
                    if z_nat_warped_sample is not None:
                        mw      = sp_mask[:, :z_nat_warped_sample.shape[1]]
                        sim_w   = (F.cosine_similarity(z_hat[:, :z_nat_warped_sample.shape[1]],
                                                       z_nat_warped_sample, dim=-1) * mw).sum(1) \
                                  / mw.sum(1).float()
                        sim_warp = sim_w.mean().item()
                    else:
                        sim_warp = float("nan")
                    # Secondary: cos(z_hat, z_nat) position-aligned — for comparison only
                    sim_pos = (F.cosine_similarity(z_hat, z_nat_sample, dim=-1) * sp_mask).sum(1) \
                              / sp_mask.sum(1).float()
                    sim = sim_warp  # primary metric stored in history
                    cosine_sims[str(epoch + 1)] = sim
                    baseline_str = f" (baseline: {baseline_sim:.4f})" if baseline_sim is not None else ""
                    print(f"  Cosine sim vs z_nat_warped (primary): {sim_warp:.4f}{baseline_str}")
                    print(f"  Cosine sim vs z_nat pos-aligned (ref): {sim_pos.mean().item():.4f}")
                else:
                    sp_mask = (torch.arange(L, device=device).unsqueeze(0)
                               < nat_speech_end_sample.unsqueeze(1))  # [B, L]
                    sim_per = (F.cosine_similarity(z_hat, z_nat_sample, dim=-1) * sp_mask).sum(1) \
                              / sp_mask.sum(1).float()
                    sim = sim_per.mean().item()
                    cosine_sims[str(epoch + 1)] = sim
                    baseline_str = f" (baseline: {baseline_sim:.4f})" if baseline_sim is not None else ""
                    print(f"  Cosine sim (speech, z_hat vs z_nat): {sim:.4f}{baseline_str}")

                # Scale check: mean per-frame L2 norm, speech and tail separately
                def _mean_norm(z, mask):
                    return (z.norm(dim=-1) * mask).sum(1).div(mask.sum(1).float()).mean().item()

                tl_mask = ~sp_mask
                print(f"  Scale speech — z_acc:{_mean_norm(z_acc_sample, sp_mask):.3f}"
                      f"  z_nat:{_mean_norm(z_nat_sample, sp_mask):.3f}"
                      f"  z_hat:{_mean_norm(z_hat, sp_mask):.3f}")
                print(f"  Scale tail   — z_acc:{_mean_norm(z_acc_sample, tl_mask):.3f}"
                      f"  z_nat:{_mean_norm(z_nat_sample, tl_mask):.3f}"
                      f"  z_hat:{_mean_norm(z_hat, tl_mask):.3f}")

        # Scheduler step
        scheduler.step()

        # Save latest checkpoint (always, for resumption)
        save_checkpoint(ckpt_latest, model, optimizer, scheduler, epoch, best_val_loss)

        # Save best checkpoint + early stopping counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_best = out_dir / "checkpoint_best.pt"
            save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch, best_val_loss)
            print(f"  ✓ New best val loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[Train] Early stopping at epoch {epoch+1} (patience={patience} exceeded)")
                break

    # Save training history and plot
    save_history(out_dir / "history.json", train_losses, val_losses,
                 train_speech_losses=train_speech_losses, train_tail_losses=train_tail_losses,
                 val_speech_losses=val_speech_losses, val_tail_losses=val_tail_losses,
                 cosine_sims=cosine_sims if cosine_sims else None)
    plot_losses(out_dir / "losses.png", train_losses, val_losses)

    print(f"\n[Train] Complete. Best val loss: {best_val_loss:.6f}")
    print(f"[Train] Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train E2 Latent Diffusion Bridge")
    parser.add_argument(
        "--mapping_train_path",
        type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
        help="Path to training mapping JSON",
    )
    parser.add_argument(
        "--mapping_val_path",
        type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
        help="Path to validation mapping JSON",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (defaults to models/bridge or models/bridge_dtw)",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        default="position",
        choices=["position", "position_fixed", "dtw", "dtw_l2pad", "dtw_engpad"],
        help="Forward process alignment: position (full interp), position_fixed (fixed L2 tail), dtw (L2 tail), dtw_l2pad (L2 silence tail), dtw_engpad (English silence tail)",
    )
    parser.add_argument(
        "--dtw_cache",
        type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/dtw_cache/dtw_paths.pkl",
        help="Path to precomputed DTW paths pickle (only used with --alignment dtw)",
    )
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sigma_max", type=float, default=2.0, help="Bridge noise scale — set to per-element std of (z_acc - z_nat)")
    parser.add_argument("--cond_acc", action="store_true", default=False, help="Condition on z_acc (I²SB cond_acc): concatenate z_acc to z_t before proj_in")
    parser.add_argument("--parameterization", type=str, default="eps", choices=["eps", "x0", "cfm", "cfm_prewarp"],
                        help="Model parameterization: eps (epsilon-prediction), x0 (direct z_nat prediction), "
                             "cfm (OT flow matching: DTW morphing timeline), "
                             "cfm_prewarp (OT flow matching: fixed L2 timeline, DTW-warped native target)")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dim")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="FFN hidden dim")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--profile", action="store_true", help="Profile first 20 batches and exit")
    parser.add_argument("--notes", type=str, default="", help="Free-text note saved to config.json")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tail_weight", type=float, default=0.3, help="Loss weight for tail frames [T_nat:] relative to speech frames [0:T_nat] (1.0 = equal)")
    parser.add_argument("--lambda_v", type=float, default=0.0,
                        help="Weight of DTW-velocity auxiliary loss (Branch B; requires --alignment dtw --parameterization x0). 0.0 = disabled.")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "models/bridge_dtw" if args.alignment == "dtw" else "models/bridge"
    train(**vars(args))
