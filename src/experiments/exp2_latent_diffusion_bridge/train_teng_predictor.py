#!/usr/bin/env python3
"""
Train a small MLP to predict T_eng (native speech end frame) from
mean-pooled z_acc speech frames + T_l2.

At bridge inference time T_eng is unknown; this predictor gives an estimate
so the N(t) mask schedule can be computed as N(t) = round(t*T_l2 + (1-t)*T_eng_hat).

Features
--------
- Mean pool of z_acc[:T_l2]    [768]
- T_l2 / T_NORM                [1]
=> input dim 769

Target
------
- T_eng / T_NORM  (scalar)

Precomputes feature cache (out_dir/features_{split}.pt) on first run.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.bridge_utils import get_split_data_dir

T_NORM = 300.0


# ---------------------------------------------------------------------------
# Feature precomputation
# ---------------------------------------------------------------------------

def _resolve_path(rel_path: str) -> Path:
    for split in ("train", "dev"):
        p = get_split_data_dir(split) / rel_path
        if p.exists():
            return p
    return None


def precompute_features(mapping_path: str, cache_path: Path, desc: str):
    with open(mapping_path) as f:
        pairs = json.load(f)

    pools, t_l2s, t_engs = [], [], []
    skipped = 0

    for pair in tqdm(pairs, desc=f"Precompute {desc}"):
        p = _resolve_path(pair["l2_encoder_state_path"])
        if p is None:
            skipped += 1
            continue

        state = torch.load(p, map_location="cpu", weights_only=False)
        z_acc = state["hidden_states"].float()  # [1500, 768]

        T_l2 = pair["l2_speech_end_frame"]
        T_l2 = max(T_l2, 1)  # guard against edge case
        pool = z_acc[:T_l2].mean(dim=0)  # [768]

        pools.append(pool)
        t_l2s.append(float(T_l2))
        t_engs.append(float(pair["nat_speech_end_frame"]))

    if skipped:
        print(f"[Precompute] Skipped {skipped} pairs with missing encoder states")

    data = {
        "pools":  torch.stack(pools),
        "t_l2s":  torch.tensor(t_l2s,  dtype=torch.float32),
        "t_engs": torch.tensor(t_engs, dtype=torch.float32),
    }
    torch.save(data, cache_path)
    print(f"[Precompute] Saved {len(pools)} items → {cache_path}")
    return data["pools"], data["t_l2s"], data["t_engs"]


def load_features(mapping_path: str, cache_path: Path, desc: str):
    if cache_path.exists():
        print(f"[Cache] Loading {desc} features from {cache_path}")
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        return d["pools"], d["t_l2s"], d["t_engs"]
    return precompute_features(mapping_path, cache_path, desc)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TEngPredictor(nn.Module):
    def __init__(self, pool_dim: int = 768, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [32]
        layers = []
        in_dim = pool_dim + 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pool: torch.Tensor, t_l2_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pool, t_l2_norm.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def linear_baseline(t_l2_tr, t_eng_tr, t_l2_dev, t_eng_dev) -> float:
    X = np.stack([t_l2_tr.numpy(), np.ones(len(t_l2_tr))], axis=1)
    a, b = np.linalg.lstsq(X, t_eng_tr.numpy(), rcond=None)[0]
    preds = a * t_l2_dev.numpy() + b
    mae = float(np.abs(preds - t_eng_dev.numpy()).mean())
    print(f"[Baseline] T_eng = {a:.4f}*T_l2 + {b:.2f}  |  dev MAE = {mae:.2f} frames")
    return mae


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load / precompute features
    pools_tr, t_l2_tr, t_eng_tr = load_features(
        args.mapping_train, out / "features_train.pt", "train")
    pools_dev, t_l2_dev, t_eng_dev = load_features(
        args.mapping_dev, out / "features_dev.pt", "dev")

    # Optional subset (applied after cache load so cache is always full dataset)
    if args.subset_frac < 1.0:
        n = int(len(pools_tr) * args.subset_frac)
        idx = torch.randperm(len(pools_tr), generator=torch.Generator().manual_seed(args.seed))[:n]
        pools_tr, t_l2_tr, t_eng_tr = pools_tr[idx], t_l2_tr[idx], t_eng_tr[idx]
        print(f"Subset: using {n}/{len(pools_tr) + len(pools_tr) - n} train items")

    print(f"Train: {len(pools_tr)}  Dev: {len(pools_dev)}")
    print(f"T_eng stats — mean: {t_eng_tr.mean():.1f}  std: {t_eng_tr.std():.1f}")

    baseline_mae = linear_baseline(t_l2_tr, t_eng_tr, t_l2_dev, t_eng_dev)

    # Normalise
    train_ds = TensorDataset(pools_tr, t_l2_tr / T_NORM, t_eng_tr / T_NORM)
    dev_ds   = TensorDataset(pools_dev, t_l2_dev / T_NORM, t_eng_dev / T_NORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"))
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=(device.type == "cuda"))

    model = TEngPredictor().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.n_epochs)
    loss_fn = nn.MSELoss()

    train_maes, dev_maes = [], []
    best_dev_mae = float("inf")

    for epoch in range(1, args.n_epochs + 1):
        # --- train ---
        model.train()
        tr_preds, tr_tgts = [], []
        for pool, t_l2_n, t_eng_n in train_loader:
            pool, t_l2_n, t_eng_n = pool.to(device), t_l2_n.to(device), t_eng_n.to(device)
            opt.zero_grad()
            pred = model(pool, t_l2_n)
            loss_fn(pred, t_eng_n).backward()
            opt.step()
            tr_preds.append(pred.detach().cpu())
            tr_tgts.append(t_eng_n.cpu())
        sched.step()
        tr_mae = (torch.cat(tr_preds) - torch.cat(tr_tgts)).abs().mean().item() * T_NORM

        # --- val ---
        model.eval()
        dev_preds = []
        with torch.no_grad():
            for pool, t_l2_n, t_eng_n in dev_loader:
                dev_preds.append(model(pool.to(device), t_l2_n.to(device)).cpu())
        dev_pred_raw = torch.cat(dev_preds) * T_NORM
        dev_mae = (dev_pred_raw - t_eng_dev).abs().mean().item()

        train_maes.append(tr_mae)
        dev_maes.append(dev_mae)
        print(f"[{epoch:3d}/{args.n_epochs}]  train MAE={tr_mae:.2f}  dev MAE={dev_mae:.2f} frames")

        if dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            torch.save(model.state_dict(), out / "model_best.pt")

    torch.save(model.state_dict(), out / "model_last.pt")

    # --- results ---
    print(f"\nLinear baseline MAE : {baseline_mae:.2f} frames")
    print(f"MLP best dev MAE    : {best_dev_mae:.2f} frames")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_maes, label="train MAE")
    ax.plot(dev_maes,   label="dev MAE")
    ax.axhline(baseline_mae, color="gray", linestyle="--", label=f"linear baseline ({baseline_mae:.1f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (frames)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "mae_curve.png", dpi=150)

    # Scatter plot (best model)
    model.load_state_dict(torch.load(out / "model_best.pt", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        final_preds = model(pools_dev.to(device),
                            (t_l2_dev / T_NORM).to(device)).cpu() * T_NORM

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_eng_dev.numpy(), final_preds.numpy(), alpha=0.15, s=5, c="steelblue")
    lim = (50, 380)
    ax.plot(lim, lim, "r--", lw=1, label="y=x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("T_eng true (frames)"); ax.set_ylabel("T_eng predicted (frames)")
    ax.set_title(f"Dev MAE = {best_dev_mae:.1f} frames  (baseline {baseline_mae:.1f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scatter_dev.png", dpi=150)

    with open(out / "config.json", "w") as f:
        json.dump({
            "t_norm": T_NORM,
            "pool_dim": 768,
            "hidden": [32],
            "subset_frac": args.subset_frac,
            "n_epochs": args.n_epochs,
            "baseline_mae_frames": round(baseline_mae, 2),
            "best_dev_mae_frames": round(best_dev_mae, 2),
        }, f, indent=2)

    print(f"Saved to {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mapping_train", default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json")
    p.add_argument("--mapping_dev",   default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json")
    p.add_argument("--out_dir",       default="models/teng_predictor")
    p.add_argument("--subset_frac",   type=float, default=1.0,
                   help="Fraction of train set to use (1.0 = full, 0.2 = 20%%)")
    p.add_argument("--n_epochs",      type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--seed",          type=int,   default=42)
    train(p.parse_args())
