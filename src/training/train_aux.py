"""
train_aux.py
Training loop for WhisperWithAuxHeads.

Ablation conditions via --lambda_ctc / --lambda_feat:
    baseline  :  0.0,  0.0
    ctc_only  :  0.3,  0.0
    feat_only :  0.0,  0.1
    both      :  0.3,  0.1

Usage:
    python train_aux.py --run_name ctc_only --lambda_ctc 0.3 --output_dir models/aux_ctc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import LOCAL_L2ARCTIC_DIR, MODEL_ID
from src.utils.load_l2arctic import load_train_dev_utterances
from src.utils.textgrid import parse_textgrid
from src.utils.phonology import PHON_FEATURE_MATRIX
from src.training.whisper_aux import WhisperWithAuxHeads

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LORA_CONFIG = LoraConfig(
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj"],
    inference_mode = False,
)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class AuxCollator:
    def __init__(
        self,
        processor:     WhisperProcessor,
        lambda_feat:   float = 0.0,
        max_label_len: int   = 448,
    ):
        self.processor     = processor
        self.lambda_feat   = lambda_feat
        self.max_label_len = max_label_len
        self.pad_id        = processor.tokenizer.pad_token_id

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = []
        for it in items:
            audio, sr = sf.read(it["wav_path"], dtype="float32", always_2d=False)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audios.append(audio)

        input_features = self.processor(
            audios, sampling_rate=16000, return_tensors="pt", padding="max_length",
        ).input_features

        labels = self.processor.tokenizer(
            [it["text"] for it in items],
            return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_label_len,
        ).input_ids.clone()
        labels[labels == self.pad_id] = -100

        T_enc             = input_features.shape[-1] // 2
        B                 = len(items)
        ctc_input_lengths = torch.full((B,), T_enc, dtype=torch.long)

        ctc_seqs, ctc_lengths, feat_targets_list, feat_frame_spans = [], [], [], []

        for it in items:
            try:
                segs = parse_textgrid(it["textgrid"])
            except Exception:
                segs = []

            phone_ids, spans, feat_vecs = [], [], []
            for seg in segs:
                if seg.phone_id < 0:
                    continue
                phone_ids.append(seg.phone_id)
                spans.append((seg.start_frame, seg.end_frame))
                if self.lambda_feat > 0:
                    feat_vecs.append(PHON_FEATURE_MATRIX[seg.phone_id])

            if not phone_ids:
                phone_ids = [0]
                spans     = [(0, 1)]
                if self.lambda_feat > 0:
                    feat_vecs = [PHON_FEATURE_MATRIX[0]]

            ctc_seqs.append(torch.tensor(phone_ids, dtype=torch.long))
            ctc_lengths.append(len(phone_ids))
            feat_frame_spans.append(spans)
            if self.lambda_feat > 0:
                feat_targets_list.extend(feat_vecs)

        batch = {
            "input_features":     input_features,
            "labels":             labels,
            "ctc_targets":        torch.cat(ctc_seqs),
            "ctc_input_lengths":  ctc_input_lengths,
            "ctc_target_lengths": torch.tensor(ctc_lengths, dtype=torch.long),
            "feat_frame_spans":   feat_frame_spans,
        }
        if self.lambda_feat > 0 and feat_targets_list:
            batch["feat_targets"] = torch.tensor(
                np.stack(feat_targets_list), dtype=torch.float32
            )
        return batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class L2ArcticDataset(Dataset):
    def __init__(self, utterances): self.utterances = utterances
    def __len__(self):              return len(self.utterances)
    def __getitem__(self, idx):     return self.utterances[idx]


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------


def plot_loss_curve(history: list[dict], out_path: Path) -> None:
    """
    history entries have two shapes:
      - epoch-level: {"epoch": int, "train": {...}, "dev": {...}}
      - step-level:  {"step": int, "epoch": int, "train_step": {...}}
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    epoch_entries = [h for h in history if "train" in h and "dev" in h]
    step_entries  = [h for h in history if "train_step" in h]

    has_ctc  = any(h["train"].get("ctc",  0) > 0 for h in epoch_entries)
    has_feat = any(h["train"].get("feat", 0) > 0 for h in epoch_entries)
    n_panels = 1 + has_ctc + has_feat

    fig = plt.figure(figsize=(6 * n_panels, 8))
    gs  = gridspec.GridSpec(2, n_panels, hspace=0.45, wspace=0.35)

    # ── Row 0: step-level train loss (or epoch if no steps logged) ────────────
    for col, (metric, title) in enumerate(
        [("total", "Total loss")]
        + ([("ctc",  "CTC loss")]  if has_ctc  else [])
        + ([("feat", "Feat loss")] if has_feat else [])
    ):
        ax = fig.add_subplot(gs[0, col])

        if step_entries:
            steps  = [s["step"]                       for s in step_entries]
            values = [s["train_step"].get(metric, 0)  for s in step_entries]
            ax.plot(steps, values, lw=0.8, alpha=0.4, color="#2196F3", label="train (step)")

            # smoothed rolling mean
            window = max(1, len(values) // 40)
            smooth = np.convolve(values, np.ones(window)/window, mode="valid")
            ax.plot(steps[:len(smooth)], smooth, lw=1.8, color="#2196F3", label=f"train (smooth {window})")
        else:
            epochs     = [h["epoch"]              for h in epoch_entries]
            train_vals = [h["train"].get(metric,0) for h in epoch_entries]
            ax.plot(epochs, train_vals, "o-", lw=2, color="#2196F3", label="train")

        ax.set_title(f"{title} — train"); ax.set_xlabel("Step" if step_entries else "Epoch")
        ax.set_ylabel("Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Row 1: epoch-level train vs dev (always available) ────────────────────
    epochs = [h["epoch"] for h in epoch_entries]
    for col, (metric, title) in enumerate(
        [("total", "Total loss")]
        + ([("ctc",  "CTC loss")]  if has_ctc  else [])
        + ([("feat", "Feat loss")] if has_feat else [])
    ):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(epochs, [h["train"].get(metric,0) for h in epoch_entries],
                "o-", lw=2, color="#2196F3", label="train")
        ax.plot(epochs, [h["dev"].get(metric,0)   for h in epoch_entries],
                "s--", lw=2, color="#FF5722", label="dev")
        ax.set_title(f"{title} — train vs dev"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Training curves", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve → {out_path}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== train_aux  run={args.run_name}  "
          f"lambda_ctc={args.lambda_ctc}  lambda_feat={args.lambda_feat}  "
          f"device={device} ===")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading utterances ...")
    train_utts, dev_utts = load_train_dev_utterances(local_root=args.data_root)
    print(f"  train={len(train_utts)}  dev={len(dev_utts)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    collator  = AuxCollator(processor, lambda_feat=args.lambda_feat)

    train_loader = DataLoader(
        L2ArcticDataset(train_utts), batch_size=args.batch_size,
        shuffle=True,  collate_fn=collator, pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        L2ArcticDataset(dev_utts), batch_size=args.batch_size,
        shuffle=False, collate_fn=collator, pin_memory=(device == "cuda"),
    )

    print("Building model ...")
    model = WhisperWithAuxHeads(
        model_name  = MODEL_ID,
        lambda_ctc  = args.lambda_ctc,
        lambda_feat = args.lambda_feat,
    )
    for p in model.whisper.parameters():
        p.requires_grad = False
    model.whisper = get_peft_model(model.whisper, LORA_CONFIG)
    model.whisper.print_trainable_parameters()
    model = model.to(device)

    optimizer   = AdamW(
        model.param_groups(base_lr=args.lr, head_lr=args.lr * 10),
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 10),
        num_training_steps = total_steps,
    )

    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project="whisper-l2-aux", name=args.run_name, config=vars(args))
            use_wandb = True
        except ImportError:
            print("[WARN] wandb not installed, skipping")

    best_dev_loss = float("inf")
    history       = []
    global_step   = 0

    for epoch in range(1, args.epochs + 1):

        # --- Train ---
        model.train()
        tr      = {"total": [], "asr": [], "ctc": [], "feat": []}
        pbar    = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs} [train]",
                       unit="batch", dynamic_ncols=True)

        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                input_features     = batch["input_features"],
                labels             = batch["labels"],
                ctc_targets        = batch["ctc_targets"],
                ctc_input_lengths  = batch["ctc_input_lengths"],
                ctc_target_lengths = batch["ctc_target_lengths"],
                feat_targets       = batch.get("feat_targets"),
                feat_frame_spans   = batch.get("feat_frame_spans"),
            )

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            tr["total"].append(out.loss.item())
            tr["asr"].append(out.loss_asr.item()   if out.loss_asr  else 0.0)
            tr["ctc"].append(out.loss_ctc.item()   if out.loss_ctc  else 0.0)
            tr["feat"].append(out.loss_feat.item() if out.loss_feat else 0.0)

            global_step += 1
            if global_step % 10 == 0:
                history.append({
                    "step":  global_step,
                    "epoch": epoch,
                    "train_step": {k: tr[k][-1] for k in tr},  # reuse already-appended values
                })

            pbar.set_postfix(
                total = f"{tr['total'][-1]:.3f}",
                asr   = f"{tr['asr'][-1]:.3f}",
                ctc   = f"{tr['ctc'][-1]:.3f}",
            )

        mean_tr = {k: float(np.mean(v)) for k, v in tr.items()}

        # --- Dev ---
        model.eval()
        dv   = {"total": [], "asr": [], "ctc": [], "feat": []}
        pbar = tqdm(dev_loader, desc=f"Ep {epoch}/{args.epochs} [dev]  ",
                    unit="batch", dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(
                    input_features     = batch["input_features"],
                    labels             = batch["labels"],
                    ctc_targets        = batch["ctc_targets"],
                    ctc_input_lengths  = batch["ctc_input_lengths"],
                    ctc_target_lengths = batch["ctc_target_lengths"],
                    feat_targets       = batch.get("feat_targets"),
                    feat_frame_spans   = batch.get("feat_frame_spans"),
                )
                dv["total"].append(out.loss.item())
                dv["asr"].append(out.loss_asr.item()   if out.loss_asr  else 0.0)
                dv["ctc"].append(out.loss_ctc.item()   if out.loss_ctc  else 0.0)
                dv["feat"].append(out.loss_feat.item() if out.loss_feat else 0.0)

                pbar.set_postfix(
                    total = f"{np.mean(dv['total']):.3f}",
                    asr   = f"{np.mean(dv['asr']):.3f}",
                    ctc   = f"{np.mean(dv['ctc']):.3f}",
                )

        mean_dv = {k: float(np.mean(v)) for k, v in dv.items()}

        print(
            f"Epoch {epoch:>2} summary | "
            f"train  total={mean_tr['total']:.4f}  asr={mean_tr['asr']:.4f}  ctc={mean_tr['ctc']:.4f}  feat={mean_tr['feat']:.4f} | "
            f"dev    total={mean_dv['total']:.4f}  asr={mean_dv['asr']:.4f}  ctc={mean_dv['ctc']:.4f}  feat={mean_dv['feat']:.4f}"
        )

        history.append({"epoch": epoch, "train": mean_tr, "dev": mean_dv})

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch,
                       **{f"train/{k}": v for k, v in mean_tr.items()},
                       **{f"dev/{k}":   v for k, v in mean_dv.items()}})

        if mean_dv["total"] < best_dev_loss:
            best_dev_loss = mean_dv["total"]
            ckpt = out_dir / "best"
            model.whisper.save_pretrained(str(ckpt))
            processor.save_pretrained(str(ckpt))
            torch.save(model.ctc_head.state_dict(),  out_dir / "best_ctc_head.pt")
            torch.save(model.feat_head.state_dict(), out_dir / "best_feat_head.pt")
            print(f"  ✓ Checkpoint (dev={best_dev_loss:.4f}) → {ckpt}")

    hist_path = out_dir / "history.json"
    hist_path.write_text(
        json.dumps({"run": args.run_name, "args": vars(args), "history": history},
                   indent=2, default=str)
    )
    plot_loss_curve(history, out_dir / "loss_curve.png")
    print(f"Done. History → {hist_path}")
    if use_wandb:
        import wandb; wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Train Whisper with auxiliary phonology heads")
    p.add_argument("--run_name",    default="ctc_only")
    p.add_argument("--data_root",   default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--output_dir",  default="models/aux_ctc")
    p.add_argument("--lambda_ctc",  type=float, default=0.1)
    p.add_argument("--lambda_feat", type=float, default=0.0)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--wandb",       action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()