import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_loss_curve(log_history: list[dict], out_path: Path, run_name: str = "baseline_lora") -> None:
    """Plot train/dev loss and dev WER, all on a consistent per-epoch x-axis."""
    # Bucket train loss by epoch (floored)
    from collections import defaultdict
    train_by_epoch = defaultdict(list)
    dev_epochs, dev_loss, dev_wer = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            epoch = int(entry.get("epoch", 0) + 1e-9)  # floor to int epoch
            if epoch > 0:
                train_by_epoch[epoch].append(entry["loss"])
        if "eval_loss" in entry:
            dev_epochs.append(entry["epoch"])
            dev_loss.append(entry["eval_loss"])
            if "eval_wer" in entry:
                dev_wer.append(entry["eval_wer"] * 100)  # → percentage

    train_epochs = sorted(train_by_epoch)
    train_means  = [float(np.mean(train_by_epoch[e])) for e in train_epochs]

    has_wer = len(dev_wer) == len(dev_epochs) and len(dev_wer) > 0
    n_panels = 2 if has_wer else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    axes = axes[0]

    # Loss panel
    ax = axes[0]
    ax.plot(train_epochs, train_means, "o-", color="#2196F3", label="train loss (epoch mean)")
    ax.plot(dev_epochs,   dev_loss,    "s--", color="#FF5722", label="dev loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_xticks(train_epochs)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # WER panel
    if has_wer:
        ax = axes[1]
        ax.plot(dev_epochs, dev_wer, "s-", color="#4CAF50")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("WER (%)")
        ax.set_title("Dev WER")
        ax.set_xticks(list(range(1, len(dev_epochs) + 1)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.grid(alpha=0.3)

    fig.suptitle(f"{run_name} training curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve → {out_path}")


state = json.loads(Path("models/baseline_loraft/checkpoint-10820/trainer_state.json").read_text())
plot_loss_curve(state["log_history"], Path("models/baseline_loraft/loss_curve.png"))