
"""
train_aux.py
Training loop for WhisperWithAuxHeads with CTC phoneme auxiliary loss.

Ablation conditions via --lambda_ctc / --lambda_feat:
    baseline:  0.0, 0.0
    ctc_only:  0.3, 0.0  
    feat_only: 0.0, 0.1
    both:      0.3, 0.1

Usage:
    python -m src.training.train_aux \\
        --run_name ctc_only \\
        --lambda_ctc 0.3 \\
        --output_dir models/aux_ctc \\
        [--epochs 5] [--batch_size 16] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from src.eval.probing.probe_utils import SPEAKER_L1
from src.training.whisper_aux import AuxCollator, WhisperWithAuxHeads
from src.utils.load_l2arctic import load_probe_utterances
from src.config import LOCAL_L2ARCTIC_DIR, HELD_OUT_SPEAKERS


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class L2ArcticDataset(Dataset):
    def __init__(self, utterances):
        self.utterances = utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return self.utterances[idx]


# ---------------------------------------------------------------------------
# LoRA config — keep identical to baseline for fair ablation
# ---------------------------------------------------------------------------

LORA_CONFIG = LoraConfig(
    task_type      = TaskType.SEQ_2_SEQ_LM,
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj"],
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run: {args.run_name}  (lambda_ctc={args.lambda_ctc}, lambda_feat={args.lambda_feat})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data -----------------------------------------------------------
    print("\\nLoading utterances ...")
    all_utts = load_probe_utterances(
        local_root = args.data_root,
        split      = "scripted",
        max_utts   = args.max_utts,
        speakers   = SPEAKER_L1.keys() - HELD_OUT_SPEAKERS 
    )
    # Speaker-stratified split so dev has all L1s represented
    train_dev_speakers = [u["speaker"] for u in all_utts]
    train_utts, dev_utts = train_test_split(
        all_utts,
        test_size    = 0.15,
        random_state = 42,
        stratify     = train_dev_speakers,   # preserve speaker/L1 distribution
    )

    print(f"  Train: {len(train_utts)}  Dev: {len(dev_utts)}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    collator  = AuxCollator(processor)

    train_loader = DataLoader(
        L2ArcticDataset(train_utts),
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collator,
        num_workers = 2,
        pin_memory  = (device == "cuda"),
    )
    dev_loader = DataLoader(
        L2ArcticDataset(dev_utts),
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collator,
        num_workers = 2,
        pin_memory  = (device == "cuda"),
    )

    # ---- Model ----------------------------------------------------------
    print("\\nBuilding model ...")
    model = WhisperWithAuxHeads(
        model_name  = "openai/whisper-small",
        lambda_ctc  = args.lambda_ctc,
        lambda_feat = args.lambda_feat,
    )

    # Freeze backbone, apply LoRA
    for p in model.whisper.parameters():
        p.requires_grad = False
    model.whisper = get_peft_model(model.whisper, LORA_CONFIG)
    model.whisper.print_trainable_parameters()

    # Aux head params always trainable
    for p in model.ctc_head.parameters():
        p.requires_grad = True
    for p in model.feat_head.parameters():
        p.requires_grad = True

    model = model.to(device)

    # ---- Optimiser ------------------------------------------------------
    optimizer = AdamW(
        model.param_groups(base_lr=args.lr, head_lr=args.lr * 10),
        weight_decay = 0.01,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ---- W&B (optional) ------------------------------------------------
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project="whisper-aux-heads", name=args.run_name, config=vars(args))
            use_wandb = True
        except ImportError:
            print("wandb not installed, skipping")

    # ---- Loop -----------------------------------------------------------
    best_dev_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):

        # --- Train ---
        model.train()
        train_losses = {"total": [], "asr": [], "ctc": [], "feat": []}

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                input_features     = batch["input_features"],
                labels             = batch["labels"],
                ctc_targets        = batch["ctc_targets"],
                ctc_input_lengths  = batch["ctc_input_lengths"],
                ctc_target_lengths = batch["ctc_target_lengths"],
            )

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_losses["total"].append(out.loss.item())
            train_losses["asr"].append(out.loss_asr.item()  if out.loss_asr  else 0.0)
            train_losses["ctc"].append(out.loss_ctc.item()  if out.loss_ctc  else 0.0)
            train_losses["feat"].append(out.loss_feat.item() if out.loss_feat else 0.0)

            if step % args.log_every == 0:
                print(
                    f"  Ep {epoch} | {step:4d}/{len(train_loader)}"
                    f" | total={out.loss.item():.4f}"
                    f" asr={train_losses['asr'][-1]:.4f}"
                    f" ctc={train_losses['ctc'][-1]:.4f}"
                )

        mean_train = {k: float(np.mean(v)) for k, v in train_losses.items()}

        # --- Dev ---
        model.eval()
        dev_losses = {"total": [], "asr": [], "ctc": [], "feat": []}

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(
                    input_features     = batch["input_features"],
                    labels             = batch["labels"],
                    ctc_targets        = batch["ctc_targets"],
                    ctc_input_lengths  = batch["ctc_input_lengths"],
                    ctc_target_lengths = batch["ctc_target_lengths"],
                )
                dev_losses["total"].append(out.loss.item())
                dev_losses["asr"].append(out.loss_asr.item()  if out.loss_asr  else 0.0)
                dev_losses["ctc"].append(out.loss_ctc.item()  if out.loss_ctc  else 0.0)
                dev_losses["feat"].append(out.loss_feat.item() if out.loss_feat else 0.0)

        mean_dev = {k: float(np.mean(v)) for k, v in dev_losses.items()}

        print(
            f"Epoch {epoch} | "
            f"train={mean_train['total']:.4f} (asr={mean_train['asr']:.4f} ctc={mean_train['ctc']:.4f}) | "
            f"dev={mean_dev['total']:.4f}   (asr={mean_dev['asr']:.4f} ctc={mean_dev['ctc']:.4f})"
        )

        history.append({"epoch": epoch, "train": mean_train, "dev": mean_dev})

        if use_wandb:
            import wandb
            wandb.log({
                "epoch":          epoch,
                "train/total":    mean_train["total"],
                "train/asr":      mean_train["asr"],
                "train/ctc":      mean_train["ctc"],
                "dev/total":      mean_dev["total"],
                "dev/asr":        mean_dev["asr"],
                "dev/ctc":        mean_dev["ctc"],
            })

        # --- Checkpoint --
        if mean_dev["total"] < best_dev_loss:
            best_dev_loss = mean_dev["total"]
            ckpt = out_dir / "best"
            model.whisper.save_pretrained(str(ckpt))
            processor.save_pretrained(str(ckpt))
            torch.save(model.ctc_head.state_dict(),  out_dir / "best_ctc_head.pt")
            torch.save(model.feat_head.state_dict(), out_dir / "best_feat_head.pt")
            print(f"  ✓ Checkpoint saved (dev={best_dev_loss:.4f}) -> {ckpt}")

    # ---- Persist history ------------------------------------------------
    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "history": history}, f, indent=2)
    print(f"\\nDone. History -> {history_path}")

    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",    default="ctc_only")
    p.add_argument("--data_root",   default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--output_dir",  default="models/aux_ctc")
    p.add_argument("--lambda_ctc",  type=float, default=0.3)
    p.add_argument("--lambda_feat", type=float, default=0.0)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--log_every",   type=int,   default=50)
    p.add_argument("--max_utts",    type=int,   default=None)
    p.add_argument("--wandb",       action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()