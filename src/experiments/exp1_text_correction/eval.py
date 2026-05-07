"""
Stage 1 evaluation: test MiniMDM as a token corrector in the decoding loop.
Integrates with frozen SMDM decoder to compute WER with/without correction.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

from src.experiments.exp1_text_correction.config import Exp1Config
from src.experiments.exp1_text_correction.model import create_mini_mdm
from src.utils.load_l2arctic import load_test_utterances
from transformers import WhisperProcessor
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
def correct_tokens(
    model: nn.Module,
    noisy_tokens: torch.Tensor,
    condition: torch.Tensor,
    visible_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Apply MiniMDM correction to visible tokens.

    Args:
        model: MiniMDM
        noisy_tokens: (T,) token IDs (mix of tokens and mask_token_id)
        condition: (1500, 768) Whisper encoder states
        visible_mask: (T,) bool mask indicating visible positions
        device: torch device

    Returns:
        corrected_tokens: (T,) with refined visible tokens
    """
    noisy_tokens = noisy_tokens.to(device).unsqueeze(0)  # (1, T)
    condition = condition.to(device).unsqueeze(0)  # (1, 1500, 768)
    visible_mask = visible_mask.to(device).unsqueeze(0)  # (1, T)

    logits = model(noisy_tokens, condition=condition)  # (1, T, vocab_size)

    # Replace visible positions with argmax predictions
    preds = logits.argmax(dim=-1)  # (1, T)
    corrected = noisy_tokens.clone()
    corrected[visible_mask] = preds[visible_mask]

    return corrected[0]  # (T,)


def evaluate_test_set(
    model: nn.Module,
    test_utterances,
    data_root: str = "data/processed",
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B",
    device: str = "cpu",
):
    """
    Evaluate MiniMDM on test set.
    For now, just load encoder states + decode with greedy selection from MiniMDM logits.
    (Full SMDM decoder integration deferred.)
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id or tokenizer.vocab_size

    results = []

    for utt in tqdm(test_utterances, desc="Evaluating"):
        speaker = utt["speaker"]
        split = utt["split"]
        utt_id = utt["utterance_id"]

        # Load encoder states
        pt_path = Path(data_root) / split / speaker / f"{utt_id}.pt"
        if not pt_path.exists():
            continue

        data = torch.load(pt_path, map_location=device)
        condition = data["hidden_states"].float()  # (1500, 768)

        # Tokenize reference
        ref_tokens = tokenizer.encode(utt["text"], add_special_tokens=False)
        ref_tokens = torch.tensor(ref_tokens[:256], dtype=torch.long)

        # Create fully masked input
        T = len(ref_tokens)
        masked_tokens = torch.full((T,), mask_token_id, dtype=torch.long)
        visible_mask = torch.zeros(T, dtype=torch.bool)

        # Run correction (simple: just correct once)
        corrected_tokens = correct_tokens(model, masked_tokens, condition, visible_mask, device)

        # Decode corrected tokens
        predicted_text = tokenizer.decode(corrected_tokens.cpu().tolist(), skip_special_tokens=True)

        # Compute WER
        wer = jiwer.wer(utt["text"], predicted_text)

        results.append({
            "utterance_id": utt_id,
            "speaker": speaker,
            "l1": utt["l1"],
            "reference": utt["text"],
            "prediction": predicted_text,
            "wer": wer,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="data/processed")
    args = parser.parse_args()

    device = args.device
    print(f"[eval] Using device: {device}")

    # Load checkpoint
    print(f"[eval] Loading checkpoint from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"[eval] Config: {config.to_dict()}")

    # Load test utterances
    print("[eval] Loading test utterances...")
    test_utts = load_test_utterances()
    print(f"[eval] Test utterances: {len(test_utts)}")

    # Evaluate
    print("[eval] Evaluating...")
    results = evaluate_test_set(
        model,
        test_utts,
        data_root=args.data_root,
        tokenizer_name=config.tokenizer_name,
        device=device,
    )

    # Report results
    print("\n[eval] Results:")
    print(f"  Evaluated: {len(results)} utterances")

    if results:
        wers = [r["wer"] for r in results]
        avg_wer = np.mean(wers)
        print(f"  Average WER: {avg_wer:.4f}")

        # Per-L1
        from collections import defaultdict
        by_l1 = defaultdict(list)
        for r in results:
            by_l1[r["l1"]].append(r["wer"])

        print("\n  Per-L1 WER:")
        for l1 in sorted(by_l1.keys()):
            wers_l1 = by_l1[l1]
            print(f"    {l1}: {np.mean(wers_l1):.4f} ({len(wers_l1)} utts)")


if __name__ == "__main__":
    main()
