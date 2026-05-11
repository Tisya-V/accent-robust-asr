"""
eval_whisfusion_perf.py

Clean evaluation script for Whisfusion using L2-Arctic.
Matches eval_model_perf.py style: WER + PER, CSV outputs.

Usage:
    python eval_whisfusion_perf.py
"""

from __future__ import annotations

import argparse
import re
import pickle
from pathlib import Path

import pandas as pd
import torch
import jiwer
from tqdm import tqdm
import numpy as np

from src.config import LOCAL_L2ARCTIC_DIR, MODELS_DIR
from src.utils.load_l2arctic import load_test_utterances

# --- NLTK / G2P setup ---
import os
import nltk

# Download required NLTK data if missing (to NLTK_DATA directory)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk_data_dir = os.environ.get('NLTK_DATA')
    print(f"Downloading NLTK averaged_perceptron_tagger_eng to {nltk_data_dir}...")
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)

import g2p_en
_G2P = g2p_en.G2p()


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_to_phones(text: str) -> list[str]:
    raw = _G2P(text)
    return [p.rstrip("012") for p in raw if p.strip() and p[0].isalpha()]


def utt_per(ref: str, pred: str) -> float | None:
    if not ref:
        return None
    ref_p = " ".join(text_to_phones(ref))
    pred_p = " ".join(text_to_phones(pred))
    if not ref_p:
        return None
    return float(jiwer.wer(ref_p, pred_p))


# ---------------------------------------------------------------------------
# Whisfusion wrapper
# ---------------------------------------------------------------------------

class WhisfusionWrapper:
    def __init__(self, base_model_path, adapter_path, device="cuda", batch_size=8, mask_ratio_schedule=None):
        from models.whisfusion.src.evaluation.evaluate_whisfusion import WhisfusionBenchmark

        self.device = device
        self.batch_size = batch_size
        self.mask_ratio_schedule = mask_ratio_schedule or [1.0, 0.9, 0.85, 0.8]

        self.model = WhisfusionBenchmark(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            device=device
        )

        self.model.warmup(num_iterations=2)

    # --------------------------------------------------
    # Single decode 
    # --------------------------------------------------
    @torch.inference_mode()
    def decode(self, hidden_state):
        tokenizer = self.model.tokenizer

        bos = tokenizer.bos_token_id or 0
        seq_len = 256

        target_ids = torch.full(
            (seq_len,),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        target_ids[0] = bos

        candidates, _, _ = self.model._generate_with_timing(
            target_ids,
            hidden_state.unsqueeze(0),  # keep batch dim
            n_candidates=15,
            n_steps=4,
            mask_ratio_schedule=self.mask_ratio_schedule,
        )

        best = max(candidates, key=lambda x: x["avg_confidence"])
        return best["text"].strip()

    @torch.inference_mode()
    def decode_with_internals(self, hidden_state, reference_text):
        """Decode and capture per-step token sequences and confidences."""
        tokenizer = self.model.tokenizer
        config = self.model.config
        model = self.model.model

        bos = tokenizer.bos_token_id or 0
        seq_len = 256
        n_candidates = 15
        n_steps = 4
        mask_ratio_schedule = self.mask_ratio_schedule

        # Initialize
        target_ids = torch.full(
            (seq_len,),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        target_ids[0] = bos

        mask_token_id = config.padded_vocab_size
        device = hidden_state.device

        # Setup batch
        batch_size = n_candidates
        input_for_mask = target_ids
        batch_condition = hidden_state.unsqueeze(0).expand(n_candidates, -1, -1)

        current_outputs = torch.full(
            (n_candidates, seq_len),
            mask_token_id,
            dtype=input_for_mask.dtype,
            device=device
        )
        current_outputs[:, 0] = input_for_mask[0]

        # Capture per-step data
        step_tokens_history = []
        mask_history = []

        # Generation steps
        for step in range(n_steps):
            mask_ratio = mask_ratio_schedule[step] if step < len(mask_ratio_schedule) else 0.7

            if mask_ratio > 0:
                rand_probs = torch.rand(batch_size, seq_len, device=device)
                mask_indices_batch = rand_probs < mask_ratio
                mask_indices_batch[:, 0] = False
            else:
                mask_indices_batch = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

            masked_inputs = current_outputs.clone()
            masked_inputs[mask_indices_batch] = mask_token_id

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(idx=masked_inputs, condition=batch_condition)

            if step == n_steps - 1:
                probs = torch.softmax(logits, dim=-1)
                max_probs, predicted_ids = torch.max(probs, dim=-1)
                final_confidences = max_probs
            else:
                predicted_ids = torch.argmax(logits, dim=-1)

            current_outputs = torch.where(mask_indices_batch, predicted_ids, masked_inputs)

            # Save step data
            step_tokens_history.append(current_outputs.cpu().clone())
            mask_history.append(mask_indices_batch.cpu().clone())

        # Postprocessing
        all_outputs_cpu = current_outputs.cpu()
        all_confidences_cpu = final_confidences.cpu()
        all_texts = tokenizer.batch_decode(all_outputs_cpu, skip_special_tokens=True)

        pad_id = tokenizer.pad_token_id
        valid_mask = (all_outputs_cpu != pad_id)
        masked_confidences = all_confidences_cpu * valid_mask.float()
        avg_confidences = masked_confidences.sum(dim=1) / valid_mask.sum(dim=1).float().clamp(min=1)

        # Find best candidate
        best_idx = max(range(batch_size), key=lambda i: float(avg_confidences[i]))
        best_text = all_texts[best_idx]

        # Encode reference for alignment
        ref_tokens = tokenizer.encode(norm(reference_text), add_special_tokens=False)
        ref_tokens = torch.tensor(ref_tokens[:seq_len], dtype=torch.long)

        # Return step data for best candidate
        return {
            'text': best_text.strip(),
            'step_tokens': torch.stack([h[best_idx] for h in step_tokens_history]).numpy(),
            'mask_history': torch.stack([h[best_idx] for h in mask_history]).numpy(),
            'token_confidences': all_confidences_cpu[best_idx].float().numpy(),
            'reference_tokens': ref_tokens.numpy(),
        }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def transcribe_from_hidden_states_batch(self, pt_paths):
        results = []

        # preload everything once
        print("\nPreloading hidden states into memory...")
        hidden_states_list = []
        for p in tqdm(pt_paths, desc="Loading hidden states"):
            data = torch.load(p, map_location="cpu", weights_only=True)
            hidden_states_list.append(data["hidden_states"])
        print(f"Preloaded {len(hidden_states_list)} hidden states.")


        print("\nTranscribing with Whisfusion...")
        for i in tqdm(range(0, len(hidden_states_list), self.batch_size), desc="Decoding"):
            batch = hidden_states_list[i:i+self.batch_size]

            batch = [h.to(self.device) for h in batch]

            for hs in batch:
                text = self.decode(hs)
                results.append(text)
        print(f"Decoded {len(results)} utterances.")

        return results

    def transcribe_with_internals_batch(self, dataset, pt_paths):
        """Transcribe and save decoder internals."""
        results = []
        internals_list = []

        print("\nTranscribing with Whisfusion (capturing internals)...")
        for d, pt_path in tqdm(zip(dataset, pt_paths), total=len(dataset), desc="Decoding"):
            # Load hidden states on-demand (stream, not preload)
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            hs = data["hidden_states"].to(self.device)

            result = self.decode_with_internals(hs, d["text"])
            results.append(result['text'])
            internals_list.append({
                'utterance_id': d['utterance_id'],
                'speaker': d['speaker'],
                'l1': d['l1'],
                'step_tokens': result['step_tokens'],
                'mask_history': result['mask_history'],
                'token_confidences': result['token_confidences'],
                'reference_tokens': result['reference_tokens'],
            })
        print(f"Decoded {len(results)} utterances.")

        return results, internals_list

# ---------------------------------------------------------------------------
# Load encoded features (for eval)
# ---------------------------------------------------------------------------
def build_pt_dataset(utterances, processed_root="data/processed/test"):
    pt_root = Path(processed_root)

    id_to_pt = {
        p.stem: p
        for p in pt_root.rglob("*.pt")
    }

    dataset = []
    missing = []

    for utt in utterances:
        speaker = utt["speaker"]
        utt_id = f"{speaker}_{utt['utterance_id']}"

        if utt_id in id_to_pt:
            dataset.append({
                **utt,
                "pt_path": id_to_pt[utt_id]
            })
        else:
            missing.append(utt_id)

    if missing:
        print(f"⚠️ Missing {len(missing)} .pt files")

    return dataset

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(dataset, model: WhisfusionWrapper, save_decoder_internals=False, internals_dir=None):
    rows = []

    pt_paths = [d["pt_path"] for d in dataset]

    if save_decoder_internals:
        predictions, internals_list = model.transcribe_with_internals_batch(dataset, pt_paths)

        if internals_dir:
            internals_dir = Path(internals_dir)
            internals_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving decoder internals to {internals_dir}...")
            for internals in internals_list:
                speaker = internals['speaker']
                utt_id = internals['utterance_id']
                speaker_dir = internals_dir / speaker
                speaker_dir.mkdir(parents=True, exist_ok=True)
                save_path = speaker_dir / f"{utt_id}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(internals, f)
    else:
        predictions = model.transcribe_from_hidden_states_batch(pt_paths)

    assert len(predictions) == len(dataset)

    for d, pred in zip(dataset, predictions):
        ref = norm(d["text"])
        pred_n = norm(pred)

        word_measures = jiwer.process_words(ref, pred_n) if ref else None

        rows.append({
            "utterance_id": d["utterance_id"],
            "speaker": d["speaker"],
            "l1": d["l1"],
            "wav_path": d["wav_path"],
            "domain": d["domain"],
            "text": d["text"],
            "prediction": pred,
            "reference_norm": ref,
            "prediction_norm": pred_n,
            "ref_num_words": len(ref.split()),
            "utt_wer": float(word_measures.wer) if word_measures else None,
            "utt_mer": float(word_measures.mer) if word_measures else None,
            "utt_per": utt_per(ref, pred_n),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_stage2_decoder_pt(model_dir: str | Path) -> Path:
    """Find the stage2_decoder .pt file in the model directory."""
    stage2_dir = Path(model_dir) / "stage2_decoder"
    if not stage2_dir.exists():
        raise FileNotFoundError(f"stage2_decoder directory not found: {stage2_dir}")

    pt_files = list(stage2_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {stage2_dir}")

    # Return the first (or most recent) .pt file
    pt_file = sorted(pt_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"Found stage2_decoder model: {pt_file}")
    return pt_file


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--output_dir", default="results/model_perf_comparison")
    parser.add_argument("--base_model_path", default="models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors")
    parser.add_argument("--model", default="whisfusion")
    parser.add_argument("--save_decoder_internals", action="store_true", help="Save per-step decoder internals")
    parser.add_argument("--internals_dir", default="results/e1_decoder_internals", help="Directory to save decoder internals")
    parser.add_argument("--mask_ratio_schedule", default="1.0,0.9,0.85,0.8", help="Comma-separated mask ratios per step (e.g., '1.0,0.9,0.85,0.8' or '0.9,0.7,0.5,0.3')")

    args = parser.parse_args()

    # Parse mask schedule
    mask_ratio_schedule = [float(x) for x in args.mask_ratio_schedule.split(',')]
    print(f"Using mask schedule: {mask_ratio_schedule}")

    print(f"Device: {device}")

    adapter_path = _find_stage2_decoder_pt(f"{MODELS_DIR}/{args.model}")
    # check if eval files already exist
    schedule_suffix = "_".join([str(x).replace(".", "") for x in mask_ratio_schedule])
    output_file = f"{args.model}_mask{schedule_suffix}_predictions.csv"
    output_path = Path(args.output_dir) / output_file
    if output_path.exists() and not args.save_decoder_internals:
        print(f"  [skip] {output_path} already exists — delete to re-run")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # load data
    utterances = load_test_utterances(local_root=args.data_root)
    print(f"Loaded {len(utterances)} utterances")
    dataset = build_pt_dataset(utterances)
    print(f"Using {len(dataset)} utterances with cached features")

    # load model
    model = WhisfusionWrapper(
        base_model_path=args.base_model_path,
        adapter_path=adapter_path,
        device=device,
        mask_ratio_schedule=mask_ratio_schedule
    )

    # run eval
    df = evaluate(
        dataset,
        model,
        save_decoder_internals=args.save_decoder_internals,
        internals_dir=args.internals_dir if args.save_decoder_internals else None
    )

    # save
    df.to_csv(output_path, index=False)

    # metrics
    refs = df["reference_norm"].fillna("").tolist()
    hyps = df["prediction_norm"].fillna("").tolist()

    corpus_measures = jiwer.process_words(refs, hyps)
    per = df["utt_per"].dropna().mean()

    print(f"\nWER: {corpus_measures.wer:.3f}")
    print(f"MER: {corpus_measures.mer:.3f}")
    print(f"PER: {per:.3f}")
    print(f"Saved → {output_path}")
    if args.save_decoder_internals:
        print(f"Saved internals → {args.internals_dir}")


if __name__ == "__main__":
    main()
