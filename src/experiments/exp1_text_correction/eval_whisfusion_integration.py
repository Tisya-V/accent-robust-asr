"""
Stage 1 evaluation: test MiniMDM in iterative Whisfusion decoding loop.
Integrates MiniMDM into the Whisfusion decoder's backward process.
Measures WER improvement with vs without token correction.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict

from src.experiments.exp1_text_correction.config import Exp1Config, DEFAULT_TOKENIZER_NAME
from src.experiments.exp1_text_correction.model import create_mini_mdm
from src.utils.load_l2arctic import load_test_utterances
from transformers import AutoTokenizer
import jiwer


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


@torch.no_grad()
def correct_visible_tokens(
    corrector: nn.Module,
    tokens: torch.Tensor,
    visible_mask: torch.Tensor,
    condition: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Apply MiniMDM correction to visible tokens in-place.

    Args:
        corrector: MiniMDM model
        tokens: (T,) token IDs
        visible_mask: (T,) bool, True = visible position
        condition: (1500, 768) Whisper encoder states
        device: torch device

    Returns:
        corrected tokens: (T,)
    """
    if not visible_mask.any():
        return tokens

    tokens_batch = tokens.unsqueeze(0).to(device)  # (1, T)
    condition_batch = condition.unsqueeze(0).to(device)  # (1, 1500, 768)
    visible_mask_batch = visible_mask.unsqueeze(0).to(device)  # (1, T)

    logits = corrector(tokens_batch, condition=condition_batch)  # (1, T, vocab_size)
    preds = logits.argmax(dim=-1)  # (1, T)

    corrected = tokens.clone()
    corrected[visible_mask] = preds[0, visible_mask]

    return corrected


def evaluate_test_set_with_whisfusion_loop(
    corrector: nn.Module,
    test_utterances: list,
    config: Exp1Config,
    data_root: str = "data/processed",
    device: str = "cpu",
    num_correction_steps: int = 3,
) -> dict:
    """
    Evaluate corrector in simulated Whisfusion decoding loop.

    Simulates: mask some tokens → unmask some → correct visible → repeat
    Measures WER with vs without correction.

    Args:
        corrector: MiniMDM model
        test_utterances: list of test utterances
        config: Exp1Config
        data_root: path to encoder states
        device: torch device
        num_correction_steps: number of unmask/correct cycles
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id or tokenizer.vocab_size

    results = []
    missing_count = 0

    for utt in tqdm(test_utterances, desc="Evaluating with Whisfusion loop"):
        speaker = utt["speaker"]
        split = utt["split"]
        utt_id = utt["utterance_id"]

        # Load encoder states
        pt_path = Path(data_root) / split / speaker / f"{utt_id}.pt"
        if not pt_path.exists():
            missing_count += 1
            continue

        data = torch.load(pt_path, map_location=device)
        condition = data["hidden_states"].float()  # (1500, 768)

        # Tokenize reference
        ref_text = utt["text"]
        ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
        ref_tokens = torch.tensor(ref_tokens[:config.max_length], dtype=torch.long)

        # === Baseline: without correction ===
        # Start with fully masked
        tokens_baseline = torch.full_like(ref_tokens, mask_token_id)
        visible_mask = torch.zeros_like(ref_tokens, dtype=torch.bool)

        # Simulate n steps of unmasking (greedy: unmask top-confidence positions)
        for step in range(num_correction_steps):
            # Unmask a fraction of tokens (progressively more)
            unmask_ratio = (step + 1) / num_correction_steps
            num_to_unmask = int(len(ref_tokens) * unmask_ratio)

            # Get model predictions for masked positions
            with torch.no_grad():
                tokens_batch = tokens_baseline.unsqueeze(0).to(device)
                condition_batch = condition.unsqueeze(0).to(device)
                logits = corrector(tokens_batch, condition=condition_batch)  # (1, T, vocab)

                # Get confidence scores for masked positions
                masked_mask = tokens_baseline == mask_token_id
                if masked_mask.any():
                    confidences = torch.nn.functional.softmax(logits[0], dim=-1)
                    max_confidences = confidences.max(dim=-1).values
                    max_confidences[~masked_mask] = -1  # Ignore already unmasked

                    # Unmask top-confidence positions
                    _, top_indices = torch.topk(max_confidences, min(num_to_unmask, masked_mask.sum().item()))
                    preds = logits[0].argmax(dim=-1)
                    tokens_baseline[top_indices] = preds[top_indices]
                    visible_mask[top_indices] = True

        # Decode baseline
        pred_baseline_text = tokenizer.decode(
            tokens_baseline[tokens_baseline != mask_token_id].tolist(),
            skip_special_tokens=True
        )
        wer_baseline = jiwer.wer(ref_text, pred_baseline_text)

        # === With correction ===
        tokens_corrected = torch.full_like(ref_tokens, mask_token_id)
        visible_mask = torch.zeros_like(ref_tokens, dtype=torch.bool)

        for step in range(num_correction_steps):
            # Unmask a fraction
            unmask_ratio = (step + 1) / num_correction_steps
            num_to_unmask = int(len(ref_tokens) * unmask_ratio)

            # Get predictions
            with torch.no_grad():
                tokens_batch = tokens_corrected.unsqueeze(0).to(device)
                condition_batch = condition.unsqueeze(0).to(device)
                logits = corrector(tokens_batch, condition=condition_batch)

                # Unmask top-confidence
                masked_mask = tokens_corrected == mask_token_id
                if masked_mask.any():
                    confidences = torch.nn.functional.softmax(logits[0], dim=-1)
                    max_confidences = confidences.max(dim=-1).values
                    max_confidences[~masked_mask] = -1

                    _, top_indices = torch.topk(max_confidences, min(num_to_unmask, masked_mask.sum().item()))
                    preds = logits[0].argmax(dim=-1)
                    tokens_corrected[top_indices] = preds[top_indices]
                    visible_mask[top_indices] = True

            # Apply correction to visible tokens
            tokens_corrected = correct_visible_tokens(
                corrector,
                tokens_corrected,
                visible_mask,
                condition,
                device,
            )

        # Decode corrected
        pred_corrected_text = tokenizer.decode(
            tokens_corrected[tokens_corrected != mask_token_id].tolist(),
            skip_special_tokens=True
        )
        wer_corrected = jiwer.wer(ref_text, pred_corrected_text)

        results.append({
            "utterance_id": utt_id,
            "speaker": speaker,
            "l1": utt.get("l1", "unknown"),
            "reference": ref_text,
            "pred_baseline": pred_baseline_text,
            "pred_corrected": pred_corrected_text,
            "wer_baseline": wer_baseline,
            "wer_corrected": wer_corrected,
            "wer_improvement": wer_baseline - wer_corrected,
        })

    if missing_count > 0:
        print(f"[eval_whisfusion_integration] WARNING: {missing_count} test utterances missing")

    return {
        "results": results,
        "missing_count": missing_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--num_correction_steps", type=int, default=3, help="Number of unmask/correct cycles")
    args = parser.parse_args()

    device = args.device
    print(f"[eval_whisfusion_integration] Using device: {device}")

    # Load checkpoint
    print(f"[eval_whisfusion_integration] Loading checkpoint from {args.checkpoint}")
    corrector, config = load_checkpoint(args.checkpoint, device)
    print(f"[eval_whisfusion_integration] Config: {config.to_dict()}")

    # Load test utterances
    print("[eval_whisfusion_integration] Loading test utterances...")
    test_utts = load_test_utterances()
    print(f"[eval_whisfusion_integration] Test utterances: {len(test_utts)}")

    # Evaluate
    print(f"\n[eval_whisfusion_integration] Evaluating with {args.num_correction_steps} correction steps...")
    eval_result = evaluate_test_set_with_whisfusion_loop(
        corrector,
        test_utts,
        config,
        data_root=args.data_root,
        device=device,
        num_correction_steps=args.num_correction_steps,
    )
    results = eval_result["results"]

    # Report results
    print("\n" + "="*80)
    print("WHISFUSION LOOP EVALUATION RESULTS")
    print("="*80)

    if not results:
        print("[eval_whisfusion_integration] No results (all missing)")
        return

    print(f"\nEvaluated: {len(results)} utterances")

    # Overall WER
    wer_baseline_list = [r["wer_baseline"] for r in results]
    wer_corrected_list = [r["wer_corrected"] for r in results]
    improvement_list = [r["wer_improvement"] for r in results]

    print(f"\nWER Baseline (no correction):   {np.mean(wer_baseline_list):.4f} ± {np.std(wer_baseline_list):.4f}")
    print(f"WER With correction:            {np.mean(wer_corrected_list):.4f} ± {np.std(wer_corrected_list):.4f}")
    print(f"WER Improvement (negative=good): {np.mean(improvement_list):.4f} ± {np.std(improvement_list):.4f}")

    # Percentage of utterances improved
    improved = sum(1 for r in results if r["wer_improvement"] > 0)
    print(f"Utterances improved: {improved}/{len(results)} ({100*improved/len(results):.1f}%)")

    # Per-L1 breakdown
    by_l1 = defaultdict(lambda: {"baseline": [], "corrected": [], "improvement": []})
    for r in results:
        by_l1[r["l1"]]["baseline"].append(r["wer_baseline"])
        by_l1[r["l1"]]["corrected"].append(r["wer_corrected"])
        by_l1[r["l1"]]["improvement"].append(r["wer_improvement"])

    print(f"\nPer-L1 Results:")
    for l1 in sorted(by_l1.keys()):
        metrics = by_l1[l1]
        print(f"  {l1}:")
        print(f"    Baseline WER:   {np.mean(metrics['baseline']):.4f}")
        print(f"    Corrected WER:  {np.mean(metrics['corrected']):.4f}")
        print(f"    Improvement:    {np.mean(metrics['improvement']):.4f} ({len(metrics['baseline'])} utts)")


if __name__ == "__main__":
    main()
