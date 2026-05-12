"""
Evaluate MiniMDM integrated into Whisfusion's decoding loop.
Loads Whisfusion decoder + MiniMDM corrector, compares WER with vs without correction.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
import sys
import csv

from src.experiments.exp1_text_correction.config import Exp1Config
from src.experiments.exp1_text_correction.model import create_mini_mdm
from transformers import AutoTokenizer
import jiwer


def load_test_utterances_from_pt(data_root: str) -> list:
    """Build test utterance list from .pt files in test data directory."""
    data_path = Path(data_root)
    utterances = []

    if not data_path.exists():
        print(f"[load_test_utterances_from_pt] ERROR: data_root not found: {data_path}")
        return utterances

    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name

        for pt_file in sorted(speaker_dir.glob("*.pt")):
            utterances.append({
                "speaker": speaker,
                "utterance_id": pt_file.stem,
            })

    print(f"[load_test_utterances_from_pt] Found {len(utterances)} test utterances")
    return utterances


def load_minimdm_checkpoint(checkpoint_path: str, device: str):
    """Load MiniMDM checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def load_whisfusion(whisfusion_path: str, device: str):
    """Load Whisfusion decoder model."""
    sys.path.insert(0, str(Path(__file__).parent / "../../.."))

    from models.whisfusion.src.lit_gpt.diffmodel import TransEncoder, Config
    from safetensors.torch import load_file

    # Load Whisfusion config and model
    config = Config.from_name("Diff_LLaMA_170M")
    model = TransEncoder(config).to(device)

    # Load weights
    if whisfusion_path.endswith('.safetensors'):
        weights = load_file(whisfusion_path)
        model.load_state_dict(weights, strict=False)
    else:
        weights = torch.load(whisfusion_path, map_location=device, weights_only=False)
        if isinstance(weights, dict) and 'state_dict' in weights:
            weights = weights['state_dict']
        model.load_state_dict(weights, strict=False)

    model = model.to(torch.bfloat16).eval()
    return model, config


@torch.no_grad()
def decode_with_whisfusion(
    whisfusion: nn.Module,
    condition: torch.Tensor,
    mask_ratio_schedule: list,
    tokenizer,
    device: str,
    max_length: int = 256,
) -> str:
    """Generate text using Whisfusion decoder with mask_ratio_schedule."""
    mask_token_id = tokenizer.vocab_size
    seq_len = max_length
    batch_size = 1

    # Initialize fully masked
    current_tokens = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)

    for step in range(len(mask_ratio_schedule)):
        mask_ratio = mask_ratio_schedule[step]

        # Get model predictions
        with torch.no_grad():
            logits = whisfusion(idx=current_tokens, condition=condition.unsqueeze(0).to(torch.bfloat16))
            logits = logits.to(torch.float32)

        # Randomly unmask positions based on confidence
        masked_mask = current_tokens == mask_token_id
        if masked_mask.any():
            # Get confidence scores
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
            confidences[~masked_mask] = -1  # Ignore already unmasked

            # Unmask top-confidence positions
            num_to_unmask = int(seq_len * (1 - mask_ratio))
            if num_to_unmask > 0:
                _, top_indices = torch.topk(
                    confidences[0],
                    min(num_to_unmask, masked_mask.sum().item()),
                    largest=True
                )
                preds = logits[0].argmax(dim=-1)
                current_tokens[0, top_indices] = preds[top_indices]

    # Decode to text
    text = tokenizer.decode(
        current_tokens[0][current_tokens[0] != mask_token_id].tolist(),
        skip_special_tokens=True
    )
    return text


@torch.no_grad()
def decode_with_correction(
    whisfusion: nn.Module,
    corrector: nn.Module,
    condition: torch.Tensor,
    mask_ratio_schedule: list,
    tokenizer,
    device: str,
    max_length: int = 256,
) -> str:
    """Generate with Whisfusion + MiniMDM correction at each step."""
    mask_token_id = tokenizer.vocab_size
    seq_len = max_length
    batch_size = 1

    current_tokens = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)

    for step in range(len(mask_ratio_schedule)):
        mask_ratio = mask_ratio_schedule[step]

        # Get Whisfusion predictions
        with torch.no_grad():
            logits_wf = whisfusion(idx=current_tokens, condition=condition.unsqueeze(0).to(torch.bfloat16))
            logits_wf = logits_wf.to(torch.float32)

        masked_mask = current_tokens == mask_token_id
        if masked_mask.any():
            confidences = torch.softmax(logits_wf, dim=-1).max(dim=-1).values
            confidences[~masked_mask] = -1

            num_to_unmask = int(seq_len * (1 - mask_ratio))
            if num_to_unmask > 0:
                _, top_indices = torch.topk(
                    confidences[0],
                    min(num_to_unmask, masked_mask.sum().item()),
                    largest=True
                )
                preds = logits_wf[0].argmax(dim=-1)
                current_tokens[0, top_indices] = preds[top_indices]

        # Apply MiniMDM correction to visible tokens
        visible_mask = current_tokens != mask_token_id
        if visible_mask.any():
            with torch.no_grad():
                logits_corr = corrector(current_tokens, condition=condition.unsqueeze(0))

            preds_corr = logits_corr[0].argmax(dim=-1)
            current_tokens[0, visible_mask[0]] = preds_corr[visible_mask[0]]

    text = tokenizer.decode(
        current_tokens[0][current_tokens[0] != mask_token_id].tolist(),
        skip_special_tokens=True
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrector_checkpoint", type=str, required=True,
                        help="Path to MiniMDM checkpoint")
    parser.add_argument("--whisfusion_checkpoint", type=str,
                        default="models/whisfusion_finetuned/stage2_decoder/stage2_decoder.pt",
                        help="Path to Whisfusion checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="data/processed/test")
    parser.add_argument("--mask_ratio_schedule", type=str, default="0.9,0.7,0.5,0.3",
                        help="Comma-separated mask ratios per step")
    parser.add_argument("--output_dir", type=str, default="results/experiment1_stage1/eval_integration",
                        help="Output directory for results")
    parser.add_argument("--skip_correction", action="store_true",
                        help="Skip correction to debug (baseline only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"[eval_integration] Using device: {device}")

    # Parse mask schedule
    mask_ratio_schedule = [float(x.strip()) for x in args.mask_ratio_schedule.split(',')]
    print(f"[eval_integration] Mask ratio schedule: {mask_ratio_schedule}")

    # Load models
    print(f"\n[eval_integration] Loading MiniMDM from {args.corrector_checkpoint}")
    corrector, config = load_minimdm_checkpoint(args.corrector_checkpoint, device)

    print(f"[eval_integration] Loading Whisfusion from {args.whisfusion_checkpoint}")
    whisfusion, wf_config = load_whisfusion(args.whisfusion_checkpoint, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)

    # Load test utterances
    print(f"[eval_integration] Loading test utterances...")
    test_utts = load_test_utterances_from_pt(args.data_root)

    # Evaluate
    print(f"\n[eval_integration] Evaluating {len(test_utts)} utterances...")
    results = []
    missing_count = 0
    debug_samples = []

    for utt_idx, utt in enumerate(tqdm(test_utts, desc="Evaluating")):
        speaker = utt["speaker"]
        utt_id = utt["utterance_id"]

        pt_path = Path(args.data_root) / speaker / f"{utt_id}.pt"
        if not pt_path.exists():
            missing_count += 1
            continue

        data = torch.load(pt_path, map_location=device, weights_only=False)
        condition = data["hidden_states"].float().to(device)  # (1500, 768)

        ref_text = data.get("transcript") or data.get("text")
        if not ref_text:
            missing_count += 1
            continue

        # Generate with Whisfusion only
        pred_baseline = decode_with_whisfusion(
            whisfusion, condition, mask_ratio_schedule, tokenizer, device, config.max_length
        )

        # Generate with Whisfusion + correction
        pred_corrected = decode_with_correction(
            whisfusion, corrector, condition, mask_ratio_schedule, tokenizer, device, config.max_length
        )

        # Compute WER
        wer_baseline = jiwer.wer(ref_text, pred_baseline)
        wer_corrected = jiwer.wer(ref_text, pred_corrected)

        result = {
            "utterance_id": utt_id,
            "speaker": speaker,
            "reference": ref_text,
            "pred_baseline": pred_baseline,
            "pred_corrected": pred_corrected,
            "wer_baseline": wer_baseline,
            "wer_corrected": wer_corrected,
            "wer_improvement": wer_baseline - wer_corrected,
        }
        results.append(result)

        # Collect debug samples (first 3)
        if len(debug_samples) < 3:
            debug_samples.append(result)

    if missing_count > 0:
        print(f"[eval_integration] WARNING: {missing_count} utterances missing")

    # Print debug samples
    print("\n" + "="*80)
    print("DEBUG: Sample predictions (first 3 utterances)")
    print("="*80)
    for i, sample in enumerate(debug_samples):
        print(f"\n[Sample {i+1}] {sample['utterance_id']} ({sample['speaker']})")
        print(f"Reference: {sample['reference'][:100]}...")
        print(f"Baseline:  {sample['pred_baseline'][:100]}...")
        print(f"Corrected: {sample['pred_corrected'][:100]}...")
        print(f"WER baseline: {sample['wer_baseline']:.4f} | WER corrected: {sample['wer_corrected']:.4f}")
        print(f"Baseline tokens: {len(sample['pred_baseline'].split())} | Corrected tokens: {len(sample['pred_corrected'].split())}")

    # Save results to CSV
    csv_path = output_dir / "eval_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "utterance_id", "speaker", "reference", "pred_baseline", "pred_corrected",
            "wer_baseline", "wer_corrected", "wer_improvement"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "utterance_id": r["utterance_id"],
                "speaker": r["speaker"],
                "reference": r["reference"],
                "pred_baseline": r["pred_baseline"],
                "pred_corrected": r["pred_corrected"],
                "wer_baseline": f"{r['wer_baseline']:.4f}",
                "wer_corrected": f"{r['wer_corrected']:.4f}",
                "wer_improvement": f"{r['wer_improvement']:.4f}",
            })
    print(f"\nResults saved to: {csv_path}")

    # Report results
    print("\n" + "="*80)
    print("WHISFUSION + MINIMDM INTEGRATION RESULTS")
    print("="*80)
    print("\nNOTE: This uses simplified single-pass decoding, not Whisfusion's multi-candidate")
    print("selection. Baseline WER may differ from official Whisfusion evaluation.")

    if not results:
        print("[eval_integration] No results")
        return

    print(f"\nEvaluated: {len(results)} utterances")

    wer_baseline_list = [r["wer_baseline"] for r in results]
    wer_corrected_list = [r["wer_corrected"] for r in results]
    improvement_list = [r["wer_improvement"] for r in results]

    print(f"\nWER Whisfusion (baseline):       {np.mean(wer_baseline_list):.4f} ± {np.std(wer_baseline_list):.4f}")
    print(f"WER Whisfusion + MiniMDM:       {np.mean(wer_corrected_list):.4f} ± {np.std(wer_corrected_list):.4f}")
    print(f"WER Improvement (negative=good): {np.mean(improvement_list):.4f} ± {np.std(improvement_list):.4f}")

    improved = sum(1 for r in results if r["wer_improvement"] > 0)
    print(f"Utterances improved: {improved}/{len(results)} ({100*improved/len(results):.1f}%)")

    # Per-speaker breakdown
    by_speaker = defaultdict(lambda: {"baseline": [], "corrected": [], "improvement": []})
    for r in results:
        by_speaker[r["speaker"]]["baseline"].append(r["wer_baseline"])
        by_speaker[r["speaker"]]["corrected"].append(r["wer_corrected"])
        by_speaker[r["speaker"]]["improvement"].append(r["wer_improvement"])

    print(f"\nPer-speaker Results (top/bottom 5):")
    improvements = {sp: np.mean(metrics['improvement']) for sp, metrics in by_speaker.items()}
    sorted_speakers = sorted(improvements.items(), key=lambda x: x[1], reverse=True)

    print(f"  Most improved:")
    for sp, imp in sorted_speakers[:5]:
        print(f"    {sp}: {imp:.4f}")

    print(f"  Least improved:")
    for sp, imp in sorted_speakers[-5:]:
        print(f"    {sp}: {imp:.4f}")


if __name__ == "__main__":
    main()
