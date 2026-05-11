"""
Stage 1 training: pretraining on synthetic perturbations.
Trains MiniMDM to correct noisy visible tokens in masked sequences.
Mirrors Whisfusion's training pattern.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict
import csv

from src.experiments.exp1_text_correction.config import Exp1Config
from src.experiments.exp1_text_correction.model import create_mini_mdm
from src.experiments.exp1_text_correction.data import create_dataloaders
from src.utils.perturb_phonemes import PhonemePerturber
from transformers import AutoTokenizer


def forward_process(
    masked_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
):
    """Apply forward process: perturb visible (non-masked) tokens."""
    if perturber is None:
        noisy_ids = masked_ids.clone()
        perturb_mask = torch.zeros_like(masked_ids, dtype=torch.bool)
        return noisy_ids, perturb_mask

    noisy_ids = masked_ids.clone()
    visible_mask = ~mask_indices

    visible_tokens = target_ids[visible_mask].unsqueeze(0)
    perturbed_visible, perturb_flat = perturber.perturb(
        visible_tokens,
        perturb_prob=perturb_prob,
        mask_token_id=int(perturber.tokenizer.mask_token_id or perturber.tokenizer.vocab_size),
    )

    noisy_ids[visible_mask] = perturbed_visible[0]

    perturb_mask = torch.zeros_like(masked_ids, dtype=torch.bool)
    perturb_mask[visible_mask] = perturb_flat[0]

    return noisy_ids, perturb_mask


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device: str,
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
    metric_interval: int = 50,
) -> Dict[str, float]:
    """Run one training epoch. Detailed metrics computed every metric_interval batches."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        condition = batch["condition"].to(device)
        target_ids = batch["target_ids"].to(device)
        masked_ids = batch["masked_ids"].to(device)
        mask_indices = batch["mask_indices"].to(device)

        noisy_ids, perturb_mask = forward_process(masked_ids, target_ids, mask_indices, perturber, perturb_prob)

        logits = model(noisy_ids, condition=condition)
        visible_mask = ~mask_indices
        B, T, V = logits.shape
        logits_visible = logits[visible_mask]
        target_visible = target_ids[visible_mask]
        loss = nn.functional.cross_entropy(logits_visible.view(-1, V), target_visible.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds_visible = logits[visible_mask].argmax(dim=-1)
            acc = (preds_visible == target_visible).float().mean()
            total_acc += acc.item()

            # Detailed metrics every N batches
            acc_perturbed = 0.0
            if (batch_idx + 1) % metric_interval == 0:
                perturb_at_visible = perturb_mask[visible_mask]
                if perturb_at_visible.any():
                    acc_perturbed = (preds_visible[perturb_at_visible] == target_visible[perturb_at_visible]).float().mean().item()

        total_loss += loss.item()
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"}
        if acc_perturbed > 0:
            postfix["acc_pert"] = f"{acc_perturbed:.4f}"
        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return {"loss": avg_loss, "acc": avg_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: str,
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation"):
        condition = batch["condition"].to(device)
        target_ids = batch["target_ids"].to(device)
        masked_ids = batch["masked_ids"].to(device)
        mask_indices = batch["mask_indices"].to(device)

        noisy_ids, _ = forward_process(masked_ids, target_ids, mask_indices, perturber, perturb_prob)

        logits = model(noisy_ids, condition=condition)
        visible_mask = ~mask_indices
        _, _, V = logits.shape
        logits_visible = logits[visible_mask]
        target_visible = target_ids[visible_mask]
        loss = nn.functional.cross_entropy(logits_visible.view(-1, V), target_visible.view(-1))

        preds = logits_visible.argmax(dim=-1)
        acc = (preds == target_visible).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return {"loss": avg_loss, "acc": avg_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Exp1Config.from_json(args.config)
    print(f"[train] Loaded config from {args.config}")

    device = args.device
    print(f"[train] Using device: {device}")

    print("[train] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        data_root=config.data_root,
        mask_ratio_range=(config.mask_ratio_low, config.mask_ratio_high),
        num_workers=config.num_workers,
    )
    print(f"[train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[train] Creating MiniMDM...")
    model = create_mini_mdm(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model size: {num_params / 1e6:.1f}M parameters")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    perturber = None
    if config.use_perturbation:
        perturber = PhonemePerturber(tokenizer, k=config.perturber_k)
        perturber.to(device)
        print(f"[train] Loaded PhonemePerturber (k={config.perturber_k})")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    results_dir = Path(config.results_dir) / Path(args.config).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    config.save(results_dir / "config.json")

    csv_path = results_dir / "results.csv"
    csv_headers = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    csv_writer.writeheader()

    # Warmup
    print("[train] Warming up...")
    model.train()
    batch = next(iter(train_loader))
    condition = batch["condition"].to(device)
    masked_ids = batch["masked_ids"].to(device)
    mask_indices = batch["mask_indices"].to(device)
    target_ids = batch["target_ids"].to(device)

    for _ in tqdm(range(5), desc="Warming up"):
        noisy_ids, _ = forward_process(masked_ids, target_ids, mask_indices, perturber, config.perturb_prob)
        logits = model(noisy_ids, condition=condition)
        visible_mask = ~mask_indices
        _, _, V = logits.shape
        logits_visible = logits[visible_mask]
        target_visible = target_ids[visible_mask]
        loss = nn.functional.cross_entropy(logits_visible.view(-1, V), target_visible.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_val_loss = float("inf")
    print(f"[train] Results directory: {results_dir}")

    for epoch in range(config.max_epochs):
        print(f"\n[train] Epoch {epoch+1}/{config.max_epochs}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            perturber=perturber,
            perturb_prob=config.perturb_prob,
            metric_interval=50,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            perturber=perturber,
            perturb_prob=config.perturb_prob,
        )
        scheduler.step()

        print(f"[train] Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"[train] Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")

        csv_writer.writerow({
            "epoch": epoch,
            "train_loss": train_metrics['loss'],
            "train_acc": train_metrics['acc'],
            "val_loss": val_metrics['loss'],
            "val_acc": val_metrics['acc'],
        })
        csv_file.flush()

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = results_dir / "checkpoint.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"[train] Saved checkpoint to {checkpoint_path}")

    csv_file.close()
    print("\n[train] Done!")
    print(f"[train] Results saved to {results_dir}")


if __name__ == "__main__":
    main()
