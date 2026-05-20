#!/usr/bin/env python3
"""
Training loop for the latent corrector (Exp2 feasibility).

Trains a plain Transformer encoder to map z_acc -> z_nat using MSE loss
on speech frames only. Target is always the raw z_nat (no warping).

Usage:
    python -m src.experiments.exp2_transformer_bridge.train \
        --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json \
        --mapping_val_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json \
        --out_dir models/corrector_position
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np  # keep for worker_init seeding
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import LatentCorrectorTransformer
from src.experiments.exp2_latent_diffusion_bridge.dataset import BridgeDataset



def collate_fn(batch):
    z_accs, z_nats, speech_ends = zip(*batch)
    return (
        torch.stack(z_accs),  # bf16 from disk — smaller CPU->GPU transfer than float32
        torch.stack(z_nats),
        torch.tensor(speech_ends),
    )




def masked_cosine_sim(z_pred: torch.Tensor, z_nat: torch.Tensor, speech_end: torch.Tensor) -> float:
    """Mean cosine similarity over speech frames only."""
    _, L, _ = z_pred.shape
    mask = torch.arange(L, device=z_pred.device).unsqueeze(0) < speech_end.unsqueeze(1)  # [B, L]
    return F.cosine_similarity(z_pred[mask], z_nat[mask], dim=-1).mean().item()


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    ckpt_compiled  = any(k.startswith("_orig_mod.") for k in state)
    model_compiled = any(k.startswith("_orig_mod.") for k in model.state_dict())
    if ckpt_compiled and not model_compiled:
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    elif not ckpt_compiled and model_compiled:
        state = {"_orig_mod." + k: v for k, v in state.items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"[Checkpoint] Resumed from epoch {ckpt['epoch']}, best_val_loss={ckpt['best_val_loss']:.6f}")
    return ckpt["epoch"], ckpt["best_val_loss"]


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="Train")
    for z_acc, z_nat, speech_end in pbar:
        z_acc = z_acc.to(device, non_blocking=True)
        z_nat = z_nat.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_pred = model(z_acc)
            loss = F.mse_loss(z_pred, z_nat)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})
    return total_loss / n


def val_epoch(model, loader, device) -> tuple[float, float]:
    """Returns (avg_val_loss, avg_cosine_sim_speech_masked)."""
    model.eval()
    total_loss, total_sim, n = 0.0, 0.0, 0
    with torch.no_grad():
        for z_acc, z_nat, speech_end in tqdm(loader, desc="Val"):
            z_acc = z_acc.to(device, non_blocking=True)
            z_nat = z_nat.to(device, non_blocking=True)
            speech_end = speech_end.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_pred = model(z_acc)

            total_loss += F.mse_loss(z_pred, z_nat).item()
            total_sim += masked_cosine_sim(z_pred, z_nat, speech_end)
            n += 1
    return total_loss / n, total_sim / n


def plot_and_save(out_dir: Path, train_losses, val_losses, cosine_sims):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train")
    ax1.plot(epochs, val_losses, "r-", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss (all frames)")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    if cosine_sims:
        sim_epochs = [int(e) for e in cosine_sims]
        sim_vals = [cosine_sims[str(e)] for e in sim_epochs]
        ax2.plot(sim_epochs, sim_vals, "g-o")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cosine Similarity (speech frames)")
        ax2.set_title("Val Cosine Sim (speech-masked)"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_dir / 'losses.png'}")


def profile_run(model, loader, optimizer, device, n_batches: int = 30):
    """Time first n_batches in detail to identify the bottleneck."""
    import time
    model.train()
    t_load = t_dev = t_fwd = t_bwd = t_opt = 0.0
    t_wall = time.time()
    it = iter(loader)
    for i in range(n_batches):
        t = time.time()
        z_acc, z_nat, speech_end = next(it)
        t_load += time.time() - t

        t = time.time()
        z_acc = z_acc.to(device, non_blocking=True)
        z_nat = z_nat.to(device, non_blocking=True)
        speech_end = speech_end.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_dev += time.time() - t

        optimizer.zero_grad()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_pred = model(z_acc)
            loss = F.mse_loss(z_pred, z_nat)
        torch.cuda.synchronize()
        t_fwd += time.time() - t

        t = time.time()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize()
        t_bwd += time.time() - t

        t = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t_opt += time.time() - t

    total = time.time() - t_wall
    print(f"\n[Profile] {n_batches} batches in {total:.1f}s  ({n_batches/total:.2f} it/s)")
    print(f"  Extrapolated epoch: {1610 * total / n_batches / 60:.1f} min")
    for label, t in [("load", t_load), ("to_device", t_dev), ("forward", t_fwd),
                     ("backward", t_bwd), ("optimizer", t_opt)]:
        print(f"  {label:12s}: {t:.2f}s  ({100*t/total:.0f}%)")
    other = total - t_load - t_dev - t_fwd - t_bwd - t_opt
    print(f"  {'other':12s}: {other:.2f}s  ({100*other/total:.0f}%)")
    if device.type == "cuda":
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB alloc, "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")


def train(
    mapping_train_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
    mapping_val_path: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
    out_dir: str = "models/corrector_position",
    profile: bool = False,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    d_model: int = 256,
    n_layers: int = 4,
    dim_feedforward: int = 1024,
    num_workers: int = 4,
    patience: int = 5,
    seed: int = 42,
    notes: str = "",
):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BridgeDataset(mapping_train_path, split="train")
    val_dataset = BridgeDataset(mapping_val_path, split="dev")
    print(f"[Train] Train={len(train_dataset)}, Val={len(val_dataset)}")

    def worker_init(wid):
        random.seed(seed + wid); np.random.seed(seed + wid); torch.manual_seed(seed + wid)

    loader_kwargs = dict(
        collate_fn=collate_fn, pin_memory=True, num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, worker_init_fn=worker_init,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = LatentCorrectorTransformer(
        d_model=d_model, n_layers=n_layers, n_heads=8,
        dim_feedforward=dim_feedforward, dropout=0.1,
    ).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {n_params:,} params")

    print("[Train] Compiling model...")
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    start_epoch, best_val_loss = 0, float("inf")
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if ckpt_latest.exists():
        start_epoch, best_val_loss = load_checkpoint(ckpt_latest, model, optimizer, scheduler, device)
        start_epoch += 1

    # Log baseline cosine sim before any training
    model.eval()
    with torch.no_grad():
        for z_acc, z_nat, speech_end in val_loader:
            z_acc = z_acc.to(device); z_nat = z_nat.to(device); speech_end = speech_end.to(device)
            sim_all = F.cosine_similarity(z_acc, z_nat, dim=-1).mean().item()
            sim_speech = masked_cosine_sim(z_acc, z_nat, speech_end)
            print(f"[Baseline] cos_sim all_frames={sim_all:.4f}  speech_only={sim_speech:.4f}")
            break

    config = dict(
        n_epochs=n_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        d_model=d_model, n_layers=n_layers, dim_feedforward=dim_feedforward,
        patience=patience, seed=seed, notes=notes,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if profile:
        profile_run(model, train_loader, optimizer, device)
        return

    train_losses, val_losses, cosine_sims = [], [], {}
    history_path = out_dir / "history.json"
    if start_epoch > 0 and history_path.exists():
        with open(history_path) as f:
            hist = json.load(f)
        train_losses = hist.get("train_losses", [])
        val_losses   = hist.get("val_losses", [])
        cosine_sims  = hist.get("cosine_sims", {})
    epochs_no_improve = 0

    for epoch in range(start_epoch, n_epochs):
        print(f"\n[Epoch {epoch+1}/{n_epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_sim = val_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cosine_sims[str(epoch + 1)] = val_sim
        print(f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  val_cosine_sim={val_sim:.4f}")

        save_checkpoint(ckpt_latest, model, optimizer, scheduler, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, scheduler, epoch, best_val_loss)
            print(f"  ✓ New best val_loss={best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[Train] Early stopping at epoch {epoch+1}")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses, "cosine_sims": cosine_sims}, f, indent=2)
    plot_and_save(out_dir, train_losses, val_losses, cosine_sims)
    print(f"\n[Train] Done. Best val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_train_path", default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json")
    parser.add_argument("--mapping_val_path", default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json")
    parser.add_argument("--out_dir", default="models/corrector_position")
    parser.add_argument("--profile", action="store_true", help="Time 30 batches and exit")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()
    train(**vars(args))
