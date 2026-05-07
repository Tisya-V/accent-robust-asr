from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "trial_name",
    "status",
    "selection_metric",
    "best_val_loss",
    "best_val_wer",
    "best_val_cer",
    "best_epoch",
    "stopped_epoch",
    "min_mask_ratio",
    "max_mask_ratio",
    "perturb_prob",
    "include_perturb_in_loss",
    "model_best_path",
    "model_best_ema_path",
    "resolved_out_dir",
]


@dataclass
class TrialSpec:
    trial_name: str
    min_mask_ratio: float
    max_mask_ratio: float
    perturb_prob: float = 0.0
    include_perturb_in_loss: bool = False

    def __str__(self) -> str:
        return (
            "TrialSpec(\n"
            f"  trial_name={self.trial_name!r},\n"
            f"  min_mask_ratio={self.min_mask_ratio:.4f},\n"
            f"  max_mask_ratio={self.max_mask_ratio:.4f},\n"
            f"  perturb_prob={self.perturb_prob:.4f},\n"
            f"  include_perturb_in_loss={self.include_perturb_in_loss}\n"
            ")"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential Stage 2 hyperparameter search tuner")

    p.add_argument("--hpsearch_dir", type=str, required=True, help="Root directory for this hyperparameter search run")
    p.add_argument(
        "--trainer_script",
        type=str,
        default="src/training/train_stage2_decoder_perturbs.py",
    )

    p.add_argument("--train_data_dir", type=str, nargs="+", required=True)
    p.add_argument("--val_data_dir", type=str, nargs="+", required=True)
    p.add_argument("--pretrain_path", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--out_model_name", type=str, default="whisfusion")
    p.add_argument("--model_name", type=str, default="Diff_LLaMA_170M")
    p.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    p.add_argument("--perturber_cache_dir", type=str, default="src/utils/cache")

    p.add_argument("--num_devices", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--second_stage_lr_multiplier", type=float, default=0.5)
    p.add_argument("--lr_scaling", type=str, default="linear")
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--layer_wise_lr_decay_rate", type=float, default=0.9)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--val_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--early_stop_metric", type=str, default="loss", choices=["loss", "wer"])

    p.add_argument("--compute_wer_cer", action="store_true")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--use_layer_wise_lr_decay", action="store_true")
    p.add_argument(
        "--resume_existing",
        action="store_true",
        help="Resume an existing hpsearch directory if present",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing hpsearch directory before starting",
    )

    return p.parse_args()


def default_trials() -> list[TrialSpec]:
    return [
        # TrialSpec("T00_baseline_m70_100", 0.7, 1.0, 0.0, False),
        TrialSpec("T01_perturb_p10_m70_100", 0.7, 1.0, 0.1, True),
        TrialSpec("T02_perturb_p30_m70_100", 0.7, 1.0, 0.3, True),
        TrialSpec("T03_perturb_p50_m70_100", 0.7, 1.0, 0.5, True),
        TrialSpec("T04_perturb_p30_m10_100", 0.1, 1.0, 0.3, True),
        TrialSpec("T05_perturb_p30_m70_100_no_perturb_loss", 0.7, 1.0, 0.3, False),
    ]


def ensure_dir_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "trials").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def init_results_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def build_command(args: argparse.Namespace, trial: TrialSpec, trial_dir: Path) -> list[str]:
    cmd = [
        "fabric",
        "run",
        args.trainer_script,
        "--strategy=ddp",
        f"--devices={args.num_devices}",
        "--num_devices", str(args.num_devices),
        "--train_data_dir", *args.train_data_dir,
        "--val_data_dir", *args.val_data_dir,
        "--pretrain_path", args.pretrain_path,
        "--base_model_path", args.base_model_path,
        "--out_dir", str(trial_dir),
        "--out_model_name", args.out_model_name,
        "--model_name", args.model_name,
        "--tokenizer_name", args.tokenizer_name,
        "--trial_name", trial.trial_name,
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--second_stage_lr_multiplier", str(args.second_stage_lr_multiplier),
        "--lr_scaling", args.lr_scaling,
        "--weight_decay", str(args.weight_decay),
        "--scheduler_type", args.scheduler_type,
        "--warmup_ratio", str(args.warmup_ratio),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--ema_decay", str(args.ema_decay),
        "--layer_wise_lr_decay_rate", str(args.layer_wise_lr_decay_rate),
        "--gradient_clip_val", str(args.gradient_clip_val),
        "--precision", args.precision,
        "--val_steps", str(args.val_steps),
        "--num_workers", str(args.num_workers),
        "--early_stop_metric", args.early_stop_metric,
        "--min_mask_ratio", str(trial.min_mask_ratio),
        "--max_mask_ratio", str(trial.max_mask_ratio),
        "--save_every_n_epochs", "0",
        "--perturber_cache_dir", args.perturber_cache_dir,
        "--use_phoneme_perturber",
        "--perturb_prob", str(trial.perturb_prob),
    ]
    if args.compute_wer_cer:
        cmd.append("--compute_wer_cer")
    if args.use_ema:
        cmd.append("--use_ema")
    if args.use_layer_wise_lr_decay:
        cmd.append("--use_layer_wise_lr_decay")
    if trial.include_perturb_in_loss:
        cmd.append("--include_perturb_in_loss")
    return cmd


def choose_best_metric(payload: dict[str, Any]) -> float:
    metric = payload.get("selection_metric", "loss")
    if metric == "wer":
        return float(payload.get("best_val_wer", float("inf")))
    return float(payload.get("best_val_loss", float("inf")))


def main() -> int:
    args = parse_args()
    root = Path(args.hpsearch_dir)

    if root.exists() and any(root.iterdir()):
        if args.overwrite:
            shutil.rmtree(root)
        elif not args.resume_existing:
            raise SystemExit(
                f"hpsearch_dir already exists and is non-empty: {root}. "
                "Pass --resume_existing to continue or --overwrite to replace it."
            )

    ensure_dir_layout(root)
    manifest_path = root / "manifest.json"
    state_path = root / "tuner_state.json"
    results_csv = root / "results.csv"
    init_results_csv(results_csv)

    trials = default_trials()
    manifest = {
        "hpsearch_dir": str(root),
        "trainer_script": args.trainer_script,
        "selection_metric": args.early_stop_metric,
        "shared_args": {
            "train_data_dir": args.train_data_dir,
            "val_data_dir": args.val_data_dir,
            "pretrain_path": args.pretrain_path,
            "base_model_path": args.base_model_path,
            "out_model_name": args.out_model_name,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "num_devices": args.num_devices,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "second_stage_lr_multiplier": args.second_stage_lr_multiplier,
            "lr_scaling": args.lr_scaling,
            "weight_decay": args.weight_decay,
            "scheduler_type": args.scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "patience": args.patience,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "compute_wer_cer": args.compute_wer_cer,
            "use_layer_wise_lr_decay": args.use_layer_wise_lr_decay,
            "layer_wise_lr_decay_rate": args.layer_wise_lr_decay_rate,
            "gradient_clip_val": args.gradient_clip_val,
            "precision": args.precision,
            "val_steps": args.val_steps,
            "num_workers": args.num_workers,
            "perturber_cache_dir": args.perturber_cache_dir,
        },
        "trials": [asdict(t) for t in trials],
    }
    write_json(manifest_path, manifest)

    state = load_json(state_path, {"trials": {}, "completed_order": []})
    print(f"Starting tuner in {root}")
    print("Edit default_trials() in this file to change the hardcoded search set.")

    for trial in trials:
        trial_dir = root / "trials" / trial.trial_name
        metrics_path = trial_dir / "final_metrics.json"
        prior = state["trials"].get(trial.trial_name, {})
        if prior.get("status") == "COMPLETED" and metrics_path.exists():
            print(f"Skipping completed trial {trial.trial_name}")
            continue

        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, trial, trial_dir)
        state["trials"][trial.trial_name] = {
            "status": "RUNNING",
            "trial_dir": str(trial_dir),
            "command": cmd,
        }
        write_json(state_path, state)
        print("\n\n\n")
        print("-" * 80)
        print(f"RUNNING TRIAL {trial.trial_name}: {trial}")
        print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)


        result = subprocess.run(cmd, cwd=Path.cwd())
        if result.returncode != 0:
            state["trials"][trial.trial_name]["status"] = "FAILED"
            state["trials"][trial.trial_name]["returncode"] = result.returncode
            write_json(state_path, state)
            print(f"Trial {trial.trial_name} failed with return code {result.returncode}")
            return result.returncode

        if not metrics_path.exists():
            state["trials"][trial.trial_name]["status"] = "FAILED"
            state["trials"][trial.trial_name]["reason"] = "missing_final_metrics"
            write_json(state_path, state)
            print(f"Trial {trial.trial_name} missing final_metrics.json")
            return 2

        metrics = load_json(metrics_path, {})
        state["trials"][trial.trial_name] = {
            "status": "COMPLETED",
            "trial_dir": str(trial_dir),
            "final_metrics_path": str(metrics_path),
            "selection_value": choose_best_metric(metrics),
        }
        state["completed_order"].append(trial.trial_name)
        write_json(state_path, state)

        append_result(results_csv, {
            "trial_name": metrics.get("trial_name", trial.trial_name),
            "status": metrics.get("status", "completed"),
            "selection_metric": metrics.get("selection_metric"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_val_wer": metrics.get("best_val_wer"),
            "best_val_cer": metrics.get("best_val_cer"),
            "best_epoch": metrics.get("best_epoch"),
            "stopped_epoch": metrics.get("stopped_epoch"),
            "min_mask_ratio": metrics.get("min_mask_ratio"),
            "max_mask_ratio": metrics.get("max_mask_ratio"),
            "perturb_prob": metrics.get("perturb_prob"),
            "include_perturb_in_loss": metrics.get("include_perturb_in_loss"),
            "model_best_path": metrics.get("model_best_path"),
            "model_best_ema_path": metrics.get("model_best_ema_path"),
            "resolved_out_dir": metrics.get("resolved_out_dir"),
        })
    
        print("-" * 80)
        print(f"COMPLETED {trial.trial_name} AT {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

    completed = []
    for trial_name, info in state.get("trials", {}).items():
        if info.get("status") == "COMPLETED":
            metrics = load_json(Path(info["final_metrics_path"]), {})
            completed.append((trial_name, metrics, choose_best_metric(metrics)))

    if not completed:
        print("No completed trials found; nothing to promote.")
        return 3

    best_trial_name, best_metrics, _ = min(completed, key=lambda x: x[2])
    best_trial_path = Path(best_metrics["resolved_out_dir"])
    promote_src = best_trial_path / "model_best.pt"

    best_trial_payload = {
        "trial_name": best_trial_name,
        "selection_metric": best_metrics.get("selection_metric"),
        "selection_value": choose_best_metric(best_metrics),
        "resolved_out_dir": best_metrics.get("resolved_out_dir"),
        "best_val_loss": best_metrics.get("best_val_loss"),
        "best_val_wer": best_metrics.get("best_val_wer"),
        "best_val_cer": best_metrics.get("best_val_cer"),
        "best_epoch": best_metrics.get("best_epoch"),
        "stopped_epoch": best_metrics.get("stopped_epoch"),
        "model_best_path": best_metrics.get("model_best_path"),
        "model_best_ema_path": best_metrics.get("model_best_ema_path"),
        "promoted_source_path": str(promote_src) if promote_src.exists() else None,
    }
    write_json(root / "best_trial.json", best_trial_payload)

    if promote_src.exists():
        shutil.copy2(promote_src, root / "best_model_stage2_decoder.pt")
        print(f"Promoted best model from {promote_src}")
    else:
        print("Best trial completed but no model_best.pt found to promote.")

    print(f"Best trial: {best_trial_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
