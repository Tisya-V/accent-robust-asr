"""
Quick profiler for exp1 training to identify bottlenecks.
Runs 20 batches with torch.profiler and prints a summary.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from src.experiments.exp1_text_correction.config import Exp1Config
from src.experiments.exp1_text_correction.data import create_dataloaders
from src.utils.perturb_phonemes import PhonemePerturber
from transformers import AutoTokenizer


def profile_training():
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config = Exp1Config.from_json("src/experiments/exp1_text_correction/configs/phoneme_perturb_low.json")

    # Create dataloaders
    train_loader, _ = create_dataloaders(
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        data_root=config.data_root,
        mask_ratio_range=(config.mask_ratio_low, config.mask_ratio_high),
        num_workers=config.num_workers,
    )

    # Create perturber
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    perturber = PhonemePerturber(tokenizer, k=config.perturber_k) if config.use_perturbation else None
    if perturber:
        perturber.to(device)

    # Profile data loading and perturbation only (avoid model import circular dependency)
    print("\n" + "="*80)
    print("DATA LOADING & PERTURBATION PROFILE (20 batches)")
    print("="*80)

    times = {"batch_retrieval": [], "to_device": [], "perturb": []}

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 20:
            break

        # Batch retrieval time (not counted - already in dataloader)
        t_retrieve = time.time()

        # Move to device
        t_device = time.time()
        condition = batch["condition"].to(device)
        target_ids = batch["target_ids"].to(device)
        masked_ids = batch["masked_ids"].to(device)
        mask_indices = batch["mask_indices"].to(device)
        times["to_device"].append(time.time() - t_device)

        # Perturbation
        t_perturb = time.time()
        visible_mask = ~mask_indices
        visible_tokens = target_ids[visible_mask].unsqueeze(0)
        if perturber:
            noisy_visible, _ = perturber.perturb(visible_tokens, perturb_prob=config.perturb_prob,
                                                   mask_token_id=int(perturber.tokenizer.mask_token_id or perturber.tokenizer.vocab_size))
            noisy_ids = masked_ids.clone()
            noisy_ids[visible_mask] = noisy_visible[0]
        else:
            noisy_ids = masked_ids.clone()
        times["perturb"].append(time.time() - t_perturb)

    # Print summary
    print(f"\n{'Operation':<20} {'Avg Time (ms)':<15} {'Total Time (s)':<15}")
    print("-" * 50)
    for op, times_list in times.items():
        if times_list:
            avg_ms = (sum(times_list) / len(times_list)) * 1000
            total_s = sum(times_list)
            print(f"{op:<20} {avg_ms:>13.2f} {total_s:>13.2f}")

    total_time = sum(sum(v) for v in times.values())
    print("-" * 50)
    print(f"{'TOTAL':<20} {(total_time/20)*1000:>13.2f} {total_time:>13.2f}")
    print(f"\nAvg per batch: {(total_time/20)*1000:.1f} ms")
    print(f"\nNote: This profiles data loading + perturbation only (not model forward/backward)")


if __name__ == "__main__":
    profile_training()
