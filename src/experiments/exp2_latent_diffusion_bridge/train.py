#!/usr/bin/env python3
"""
Training loop for E2 Latent Diffusion Bridge.
Plain PyTorch with bf16 AMP, gradient clipping, and checkpoint resumption.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .model import BridgeTransformer
from .diffusion import bridge_loss
from .dataset import BridgeDataset


def collate_fn(batch):
    """Stack batch into tensors."""
    z_accs, z_nats, speech_ends = zip(*batch)
    z_acc = torch.stack(z_accs)  # [B, 1500, 768]
    z_nat = torch.stack(z_nats)  # [B, 1500, 768]
    speech_end = torch.tensor(speech_ends)  # [B]
    return z_acc, z_nat, speech_end


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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma_max: float = 0.5,
    profile: bool = False,
) -> float:
    """Train for one epoch. Returns average loss."""
    import time

    model.train()
    total_loss = 0.0
    num_batches = 0

    # Timing buckets
    time_load = 0.0
    time_fwd = 0.0
    time_bwd = 0.0
    time_opt = 0.0

    t_batch_start = time.time()
    iterator = iter(train_loader)
    pbar = tqdm(total=len(train_loader), desc="Training")

    while num_batches < len(train_loader):
        # Time data loading (includes disk I/O + tensor creation)
        t_load_start = time.time()
        try:
            z_acc, z_nat, speech_end = next(iterator)
        except StopIteration:
            break
        time_load += time.time() - t_load_start

        z_acc = z_acc.to(device, non_blocking=True)
        z_nat = z_nat.to(device, non_blocking=True)
        speech_end = speech_end.to(device, non_blocking=True)

        # Diagnostic: print latent scales on first batch of epoch
        if num_batches == 0:
            print(f"[Latent scales] z_nat: mean={z_nat.mean():.3f}, std={z_nat.std():.3f}")
            print(f"[Latent scales] z_acc: mean={z_acc.mean():.3f}, std={z_acc.std():.3f}")
            print(f"[Latent scales] diff MSE (baseline): {F.mse_loss(z_acc, z_nat):.4f}")

        optimizer.zero_grad()

        # Forward pass
        t_fwd_start = time.time()
        loss = bridge_loss(model, z_nat, z_acc, speech_end, sigma_max=sigma_max)
        time_fwd += time.time() - t_fwd_start

        # Backward
        t_bwd_start = time.time()
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        time_bwd += time.time() - t_bwd_start

        # Optimizer step
        t_opt_start = time.time()
        optimizer.step()
        time_opt += time.time() - t_opt_start

        total_loss += loss.item()
        num_batches += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if profile and num_batches >= 20:
            break

    pbar.close()

    if profile:
        total_time = time.time() - t_batch_start
        print(f"\n[Profile - first {num_batches} batches]")
        print(f"  Total:      {total_time:.2f}s ({num_batches/total_time:.2f} it/s)")
        print(f"  Load:       {time_load:.2f}s ({100*time_load/total_time:.1f}%) [disk I/O + tensor creation]")
        print(f"  Forward:    {time_fwd:.2f}s ({100*time_fwd/total_time:.1f}%)")
        print(f"  Backward:   {time_bwd:.2f}s ({100*time_bwd/total_time:.1f}%)")
        print(f"  Optimizer:  {time_opt:.2f}s ({100*time_opt/total_time:.1f}%)")

    avg_loss = total_loss / num_batches
    return avg_loss


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    sigma_max: float = 0.5,
) -> float:
    """Validate for one epoch. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        first_batch = True
        for z_acc, z_nat, speech_end in pbar:
            z_acc = z_acc.to(device, non_blocking=True)
            z_nat = z_nat.to(device, non_blocking=True)
            speech_end = speech_end.to(device, non_blocking=True)

            # Diagnostic: print latent scales on first batch of validation
            if first_batch:
                print(f"[Val latent scales] z_nat: mean={z_nat.mean():.3f}, std={z_nat.std():.3f}")
                print(f"[Val latent scales] z_acc: mean={z_acc.mean():.3f}, std={z_acc.std():.3f}")
                print(f"[Val latent scales] diff MSE (baseline): {F.mse_loss(z_acc, z_nat):.4f}")
                first_batch = False

            # Forward pass
            loss = bridge_loss(model, z_nat, z_acc, speech_end, sigma_max=sigma_max)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    mapping_train_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
    mapping_val_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
    out_dir: str = "models/bridge",
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    sigma_max: float = 0.5,
    num_workers: int = 4,
    profile: bool = False,
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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"[Train] Loading training data from {mapping_train_path}")
    train_dataset = BridgeDataset(mapping_train_path, split="train")

    print(f"[Train] Loading validation data from {mapping_val_path}")
    val_dataset = BridgeDataset(mapping_val_path, split="dev")

    print(f"[Train] Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Model
    print(f"[Train] Initializing BridgeTransformer")
    model = BridgeTransformer(d_model=768, n_layers=4, n_heads=8, dim_feedforward=2048)
    model = model.to(device)

    # Compile for speedup (disabled due to multiprocessing conflicts with torch.compile + DataLoader)
    # print(f"[Train] Compiling model...")
    # model = torch.compile(model)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    # Training loop
    print(f"[Train] Starting training for {n_epochs} epochs")
    for epoch in range(start_epoch, n_epochs):
        print(f"\n[Epoch {epoch+1}/{n_epochs}]")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, sigma_max=sigma_max, profile=profile)
        print(f"  Train loss: {train_loss:.6f}")

        # Validate
        val_loss = val_epoch(model, val_loader, device, sigma_max=sigma_max)
        print(f"  Val loss:   {val_loss:.6f}")

        # Scheduler step
        scheduler.step()

        # Save latest checkpoint (always, for resumption)
        save_checkpoint(ckpt_latest, model, optimizer, scheduler, epoch, best_val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_best = out_dir / "checkpoint_best.pt"
            save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch, best_val_loss)
            print(f"  ✓ New best val loss: {best_val_loss:.6f}")

    print(f"\n[Train] Complete. Best val loss: {best_val_loss:.6f}")
    print(f"[Train] Checkpoints saved to {out_dir}")


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
        default="models/bridge",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sigma_max", type=float, default=0.5, help="Noise schedule parameter")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--profile", action="store_true", help="Profile first 20 batches and exit")

    args = parser.parse_args()
    train(**vars(args))
