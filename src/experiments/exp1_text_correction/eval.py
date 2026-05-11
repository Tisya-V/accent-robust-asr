"""
Stage 1 evaluation: test MiniMDM token correction quality on test set.
Evaluates using token accuracy (not WER).
Measures: accuracy on perturbed vs clean tokens on the same distribution as training.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import random
from collections import defaultdict

from src.experiments.exp1_text_correction.config import Exp1Config, DEFAULT_TOKENIZER_NAME
from src.experiments.exp1_text_correction.model import create_mini_mdm
from src.experiments.exp1_text_correction.train import forward_process
from src.utils.perturb_phonemes import PhonemePerturber
from src.utils.load_l2arctic import load_test_utterances
from transformers import AutoTokenizer


def load_checkpoint(checkpoint_path: str, device: str):
    """Load a Stage 1 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]
    config = Exp1Config(**config_dict)

    model = create_mini_mdm(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, config


def evaluate_test_set(
    model: nn.Module,
    test_utterances: list,
    config: Exp1Config,
    data_root: str = "data/processed",
    device: str = "cpu",
    perturber=None,
) -> dict:
    """
    Evaluate MiniMDM on test set.
    Returns token accuracy metrics on perturbed vs clean tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id or tokenizer.vocab_size

    all_results = []
    missing_count = 0

    for utt in tqdm(test_utterances, desc="Evaluating test set"):
        speaker = utt["speaker"]
        split = utt["split"]
        utt_id = utt["utterance_id"]

        # Load encoder states: data_root/split/speaker/utt_id.pt
        pt_path = Path(data_root) / split / speaker / f"{utt_id}.pt"
        if not pt_path.exists():
            missing_count += 1
            continue

        data = torch.load(pt_path, map_location=device)
        condition = data["hidden_states"].float()  # (1500, 768)

        # Tokenize reference
        text = utt["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Truncate/pad to max_length
        if len(tokens) > config.max_length:
            tokens = tokens[:config.max_length]
        elif len(tokens) < config.max_length:
            tokens = torch.nn.functional.pad(
                tokens, (0, config.max_length - len(tokens)),
                value=tokenizer.pad_token_id
            )

        target_ids = tokens.clone()

        # Create masked input: sample mask ratio from training range
        mask_ratio = random.uniform(config.mask_ratio_low, config.mask_ratio_high)
        mask_indices = torch.rand(config.max_length) < mask_ratio
        masked_ids = tokens.clone()
        masked_ids[mask_indices] = mask_token_id

        # Forward process: perturb visible tokens
        noisy_ids, perturb_mask = forward_process(
            masked_ids.unsqueeze(0),
            target_ids.unsqueeze(0),
            mask_indices.unsqueeze(0),
            perturber=perturber,
            perturb_prob=config.perturb_prob,
        )
        noisy_ids = noisy_ids[0]
        perturb_mask = perturb_mask[0]

        # Model forward pass
        with torch.no_grad():
            condition_batch = condition.unsqueeze(0).to(device)  # (1, 1500, 768)
            noisy_batch = noisy_ids.unsqueeze(0).to(device)  # (1, max_length)
            logits = model(noisy_batch, condition=condition_batch)  # (1, max_length, vocab_size)

        # Evaluate on visible (non-masked) positions
        visible_mask = ~mask_indices
        if not visible_mask.any():
            continue

        # Get predictions on visible positions
        preds_visible = logits[0, visible_mask].argmax(dim=-1)
        target_visible = target_ids[visible_mask]
        perturb_at_visible = perturb_mask[visible_mask]

        # Compute accuracies
        acc_all = (preds_visible == target_visible).float().mean().item()

        if perturb_at_visible.any():
            acc_perturbed = (
                preds_visible[perturb_at_visible] == target_visible[perturb_at_visible]
            ).float().mean().item()
        else:
            acc_perturbed = 1.0  # No perturbed tokens

        if (~perturb_at_visible).any():
            acc_clean = (
                preds_visible[~perturb_at_visible] == target_visible[~perturb_at_visible]
            ).float().mean().item()
        else:
            acc_clean = 1.0  # No clean tokens

        all_results.append({
            "speaker": speaker,
            "l1": utt.get("l1", "unknown"),
            "utterance_id": utt_id,
            "acc_all": acc_all,
            "acc_perturbed": acc_perturbed,
            "acc_clean": acc_clean,
            "num_visible": visible_mask.sum().item(),
            "num_perturbed": perturb_at_visible.sum().item(),
        })

    if missing_count > 0:
        print(f"[eval] WARNING: {missing_count} test utterances missing from {data_root}")

    return {
        "results": all_results,
        "missing_count": missing_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="data/processed")
    args = parser.parse_args()

    device = args.device
    print(f"[eval] Using device: {device}")

    # Load checkpoint
    print(f"[eval] Loading checkpoint from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"[eval] Config: {config.to_dict()}")

    # Load tokenizer and perturber
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    perturber = None
    if config.use_perturbation:
        perturber = PhonemePerturber(tokenizer, k=config.perturber_k)
        perturber.to(device)
        print(f"[eval] Loaded PhonemePerturber (k={config.perturber_k})")

    # Load test utterances
    print("[eval] Loading test utterances...")
    test_utts = load_test_utterances()
    print(f"[eval] Test utterances: {len(test_utts)}")

    # Evaluate on test set
    print("\n[eval] Evaluating test set...")
    eval_result = evaluate_test_set(
        model,
        test_utts,
        config,
        data_root=args.data_root,
        device=device,
        perturber=perturber,
    )
    results = eval_result["results"]

    # Report results
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)

    if not results:
        print("[eval] No test results (all missing)")
        return

    print(f"\nEvaluated: {len(results)} utterances")

    # Overall accuracies
    acc_all_list = [r["acc_all"] for r in results]
    acc_perturbed_list = [r["acc_perturbed"] for r in results]
    acc_clean_list = [r["acc_clean"] for r in results]

    print(f"\nToken Accuracy:")
    print(f"  All visible tokens: {np.mean(acc_all_list):.4f} ± {np.std(acc_all_list):.4f}")
    print(f"  Perturbed tokens:   {np.mean(acc_perturbed_list):.4f} ± {np.std(acc_perturbed_list):.4f}")
    print(f"  Clean tokens:       {np.mean(acc_clean_list):.4f} ± {np.std(acc_clean_list):.4f}")

    # Per-L1 breakdown
    by_l1 = defaultdict(list)
    for r in results:
        by_l1[r["l1"]].append(r["acc_all"])

    print(f"\nPer-L1 Accuracy (all tokens):")
    for l1 in sorted(by_l1.keys()):
        accs = by_l1[l1]
        print(f"  {l1}: {np.mean(accs):.4f} ± {np.std(accs):.4f} ({len(accs)} utts)")

    # Per-speaker breakdown
    by_speaker = defaultdict(list)
    for r in results:
        by_speaker[r["speaker"]].append(r["acc_all"])

    if len(by_speaker) > 10:
        print(f"\nTop/bottom 5 speakers (by accuracy):")
        speaker_means = {sp: np.mean(accs) for sp, accs in by_speaker.items()}
        sorted_speakers = sorted(speaker_means.items(), key=lambda x: x[1])
        print(f"  Bottom 5:")
        for sp, acc in sorted_speakers[:5]:
            print(f"    {sp}: {acc:.4f}")
        print(f"  Top 5:")
        for sp, acc in sorted_speakers[-5:]:
            print(f"    {sp}: {acc:.4f}")
    else:
        print(f"\nPer-speaker Accuracy (all tokens):")
        for sp in sorted(by_speaker.keys()):
            accs = by_speaker[sp]
            print(f"  {sp}: {np.mean(accs):.4f} ± {np.std(accs):.4f} ({len(accs)} utts)")


if __name__ == "__main__":
    main()
