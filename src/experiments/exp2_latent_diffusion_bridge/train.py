#!/usr/bin/env python3
"""
Training loop for E2 Latent Diffusion Bridge.
Plain PyTorch with bf16 AMP, gradient clipping, and checkpoint resumption.
"""

import argparse
import json
import random
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
    """Stack batch into tensors (position alignment)."""
    z_accs, z_nats, l2_ends, nat_ends = zip(*batch)
    return (
        torch.stack(z_accs),        # [B, 1500, 768]
        torch.stack(z_nats),        # [B, 1500, 768]
        torch.tensor(l2_ends),      # [B]
        torch.tensor(nat_ends),     # [B]
    )


def collate_fn_dtw(batch):
    """Stack tensors and collect DTW paths as a list (variable-length, cannot stack)."""
    z_accs, z_nats, l2_ends, nat_ends, paths = zip(*batch)
    return (
        torch.stack(z_accs),        # [B, 1500, 768]
        torch.stack(z_nats),        # [B, 1500, 768]
        torch.tensor(l2_ends),      # [B]
        torch.tensor(nat_ends),     # [B]
        list(paths),                # list[np.ndarray], variable P per item
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


def save_history(history_path: Path, train_losses: list, val_losses: list, cosine_sims: dict = None):
    """Save training history to JSON."""
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if cosine_sims:
        history["cosine_similarities"] = cosine_sims
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


def _compute_loss(model, batch, device, sigma_max, alignment):
    """Dispatch to position or DTW loss based on alignment."""
    if alignment == "dtw":
        z_acc, z_nat, l2_speech_end, nat_speech_end, paths = batch
        # z_nat and z_acc intentionally kept on CPU — bridge_loss_dtw uses them
        # for numpy computation and handles device placement internally
        l2_speech_end  = l2_speech_end.to(device, non_blocking=True)
        nat_speech_end = nat_speech_end.to(device, non_blocking=True)
        return bridge_loss_dtw(model, z_nat, z_acc, l2_speech_end, nat_speech_end,
                               paths, sigma_max=sigma_max, device=device)
    else:
        z_acc, z_nat, l2_speech_end, nat_speech_end = batch
        z_acc         = z_acc.to(device, non_blocking=True)
        z_nat         = z_nat.to(device, non_blocking=True)
        l2_speech_end = l2_speech_end.to(device, non_blocking=True)
        return bridge_loss(model, z_nat, z_acc, l2_speech_end, sigma_max=sigma_max)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma_max: float = 0.5,
    alignment: str = "position",
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = _compute_loss(model, batch, device, sigma_max, alignment)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    pbar.close()
    return total_loss / num_batches


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
            z_acc, z_nat, l2_speech_end, nat_speech_end = next(iterator)
        except StopIteration:
            break
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
            loss = bridge_loss(model, z_nat, z_acc, l2_speech_end, sigma_max=sigma_max)
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
) -> float:
    """Validate for one epoch. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = _compute_loss(model, batch, device, sigma_max, alignment)
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / num_batches


def train(
    mapping_train_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
    mapping_val_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
    out_dir: str = "models/bridge",
    alignment: str = "position",
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    sigma_max: float = 0.5,
    num_workers: int = 4,
    profile: bool = False,
    notes: str = "",
    patience: int = 5,
    seed: int = 42,
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
    train_dataset = BridgeDataset(mapping_train_path, split="train", alignment=alignment)

    print(f"[Train] Loading validation data from {mapping_val_path}")
    val_dataset = BridgeDataset(mapping_val_path, split="dev", alignment=alignment)

    print(f"[Train] Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")

    def _worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    _collate = collate_fn_dtw if alignment == "dtw" else collate_fn

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
    model = BridgeTransformer(d_model=256, n_layers=4, n_heads=8, dim_feedforward=1024)
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
        "num_workers": num_workers,
        "model": "BridgeTransformer",
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "dim_feedforward": 1024,
        "patience": patience,
        "seed": seed,
        "notes": notes,
    }
    save_config(out_dir / "config.json", **config)

    # Capture a sample batch for sanity checks (cosine similarity of denoised latents)
    z_acc_sample          = None
    z_nat_sample          = None
    l2_speech_end_sample  = None
    for z_acc, z_nat, l2_speech_end, nat_speech_end in train_loader:
        z_acc_sample         = z_acc[:4].to(device, non_blocking=True)
        z_nat_sample         = z_nat[:4].to(device, non_blocking=True)
        l2_speech_end_sample = l2_speech_end[:4].to(device, non_blocking=True)
        break

    # Baseline cosine similarity: how similar are z_acc and z_nat before any bridge correction?
    baseline_sim = None
    if z_acc_sample is not None:
        mask = torch.arange(z_acc_sample.shape[1], device=device).unsqueeze(0) < l2_speech_end_sample.unsqueeze(1)
        sim_per = (F.cosine_similarity(z_acc_sample, z_nat_sample, dim=-1) * mask).sum(1) / mask.sum(1).float()
        baseline_sim = sim_per.mean().item()
        print(f"[Train] Baseline cosine sim (z_acc vs z_nat, speech frames only): {baseline_sim:.4f}")

    # Training loop with loss history
    train_losses = []
    val_losses = []
    cosine_sims = {}  # Track cosine similarities: {epoch: value}
    epochs_no_improve = 0
    print(f"[Train] Starting training for {n_epochs} epochs (patience={patience})")
    for epoch in range(start_epoch, n_epochs):
        print(f"\n[Epoch {epoch+1}/{n_epochs}]")

        # Train
        if profile:
            train_loss = train_epoch_profile(model, train_loader, optimizer, device, sigma_max=sigma_max)
            break  # Profile only runs once
        else:
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                     sigma_max=sigma_max, alignment=alignment)
        print(f"  Train loss: {train_loss:.6f}")
        train_losses.append(train_loss)

        # Validate
        val_loss = val_epoch(model, val_loader, device, sigma_max=sigma_max, alignment=alignment)
        print(f"  Val loss:   {val_loss:.6f}")
        val_losses.append(val_loss)

        # Sanity check: every 5 epochs, measure cosine similarity of denoised latents
        if (epoch + 1) % 5 == 0 and z_acc_sample is not None:
            from .diffusion import bridge_inference
            with torch.no_grad():
                z_hat = bridge_inference(model, z_acc_sample, n_steps=20, sigma_max=sigma_max)
                mask = torch.arange(z_hat.shape[1], device=device).unsqueeze(0) < l2_speech_end_sample.unsqueeze(1)
                sim_per = (F.cosine_similarity(z_hat, z_nat_sample, dim=-1) * mask).sum(1) / mask.sum(1).float()
                sim = sim_per.mean().item()
                cosine_sims[str(epoch + 1)] = sim
                baseline_str = f" (baseline: {baseline_sim:.4f})" if baseline_sim is not None else ""
                print(f"  Cosine sim (z_hat vs z_nat, speech frames only): {sim:.4f}{baseline_str}")

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
    save_history(out_dir / "history.json", train_losses, val_losses, cosine_sims=cosine_sims if cosine_sims else None)
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
        choices=["position", "dtw"],
        help="Forward process alignment: position (frame-by-frame) or dtw (alpha-timeline)",
    )
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sigma_max", type=float, default=0.5, help="Noise schedule parameter")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--profile", action="store_true", help="Profile first 20 batches and exit")
    parser.add_argument("--notes", type=str, default="", help="Free-text note saved to config.json")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "models/bridge_dtw" if args.alignment == "dtw" else "models/bridge"
    train(**vars(args))
