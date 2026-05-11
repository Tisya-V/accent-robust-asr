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

from src.experiments.exp1_text_correction.config import Exp1Config
from src.experiments.exp1_text_correction.model import create_mini_mdm
from src.experiments.exp1_text_correction.data import create_dataloaders
from src.utils.perturb_phonemes import PhonemePerturber
from transformers import AutoTokenizer


def forward_process(
    masked_ids: torch.Tensor,  # (B, T) masked input (mask_token at masked positions)
    target_ids: torch.Tensor,  # (B, T) clean tokens
    mask_indices: torch.Tensor,  # (B, T) bool, True = masked position
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
) -> torch.Tensor:
    """
    Apply forward process: perturb visible (non-masked) tokens.

    Args:
        masked_ids: input with mask tokens at masked positions
        target_ids: clean reference tokens
        mask_indices: boolean mask (True = masked position)
        perturber: PhonemePerturber instance
        perturb_prob: probability of perturbing a visible token

    Returns:
        noisy_ids: masked_ids with visible tokens optionally perturbed
    """
    noisy_ids = masked_ids.clone()

    if perturber is None:
        return noisy_ids

    # Extract visible tokens only (non-masked positions)
    visible_mask = ~mask_indices  # invert: True = visible
    visible_tokens = target_ids[visible_mask].unsqueeze(0)  # (1, num_visible)

    # Apply perturbation
    perturbed_visible, _ = perturber.perturb(
        visible_tokens,
        perturb_prob=perturb_prob,
        mask_token_id=perturber.tokenizer.mask_token_id or perturber.tokenizer.vocab_size,
    )

    # Put perturbed tokens back into noisy_ids
    noisy_ids[visible_mask] = perturbed_visible[0]

    return noisy_ids


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device: str,
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    total_visible_acc = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        condition = batch["condition"].to(device)  # (B, 1500, 768)
        target_ids = batch["target_ids"].to(device)  # (B, max_length)
        masked_ids = batch["masked_ids"].to(device)  # (B, max_length)
        mask_indices = batch["mask_indices"].to(device)  # (B, max_length)

        # Forward process: perturb visible tokens
        noisy_ids = forward_process(
            masked_ids,
            target_ids,
            mask_indices,
            perturber=perturber,
            perturb_prob=perturb_prob,
        )

        # Model forward pass
        logits = model(noisy_ids, condition=condition)  # (B, max_length, vocab_size)

        # Loss: only on visible (non-masked) positions
        visible_mask = ~mask_indices
        B, T, V = logits.shape
        logits_visible = logits[visible_mask]  # (num_visible,)
        target_visible = target_ids[visible_mask]  # (num_visible,)

        loss = nn.functional.cross_entropy(logits_visible.view(-1, V), target_visible.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds_visible = logits[visible_mask].argmax(dim=-1)
            visible_acc = (preds_visible == target_visible).float().mean()
            total_visible_acc += visible_acc.item()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "visible_acc": f"{visible_acc.item():.4f}",
        })

    avg_loss = total_loss / num_batches
    avg_visible_acc = total_visible_acc / num_batches

    return avg_loss, avg_visible_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: str,
    perturber: PhonemePerturber = None,
    perturb_prob: float = 0.15,
):
    """Run validation."""
    model.eval()
    total_loss = 0
    total_visible_acc = 0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation"):
        condition = batch["condition"].to(device)
        target_ids = batch["target_ids"].to(device)
        masked_ids = batch["masked_ids"].to(device)
        mask_indices = batch["mask_indices"].to(device)

        # Forward process
        noisy_ids = forward_process(
            masked_ids,
            target_ids,
            mask_indices,
            perturber=perturber,
            perturb_prob=perturb_prob,
        )

        logits = model(noisy_ids, condition=condition)

        visible_mask = ~mask_indices
        B, T, V = logits.shape
        logits_visible = logits[visible_mask]
        target_visible = target_ids[visible_mask]

        loss = nn.functional.cross_entropy(logits_visible.view(-1, V), target_visible.view(-1))
        preds = logits_visible.argmax(dim=-1)
        acc = (preds == target_visible).float().mean()

        total_loss += loss.item()
        total_visible_acc += acc.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_visible_acc / num_batches

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    config = Exp1Config.from_json(args.config)
    print(f"[train] Loaded config from {args.config}")
    print(f"[train] Config: {config.to_dict()}")

    device = args.device
    print(f"[train] Using device: {device}")

    # Create dataloaders
    print("[train] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        data_root=config.data_root,
        mask_ratio_range=(config.visible_mask_ratio_low, config.visible_mask_ratio_high),
    )
    print(f"[train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("[train] Creating MiniMDM...")
    model = create_mini_mdm(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model size: {num_params / 1e6:.1f}M parameters")

    # Create tokenizer and perturber
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    perturber = None
    if config.use_perturbation:
        perturber = PhonemePerturber(tokenizer, k=config.perturber_k)
        print(f"[train] Loaded PhonemePerturber (k={config.perturber_k})")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # Training loop
    results_dir = Path(config.results_dir) / Path(args.config).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    print(f"[train] Results directory: {results_dir}")

    for epoch in range(config.max_epochs):
        print(f"\n[train] Epoch {epoch+1}/{config.max_epochs}")

        train_loss, train_visible_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            perturber=perturber,
            perturb_prob=config.perturb_prob,
        )
        val_loss, val_acc = validate(
            model,
            val_loader,
            device,
            perturber=perturber,
            perturb_prob=config.perturb_prob,
        )
        scheduler.step()

        print(f"[train] Train Loss: {train_loss:.4f}, Visible Acc: {train_visible_acc:.4f}")
        print(f"[train] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = results_dir / "checkpoint.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"[train] Saved checkpoint to {checkpoint_path}")

    print("\n[train] Done!")


if __name__ == "__main__":
    main()
