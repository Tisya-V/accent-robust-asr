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
from pathlib import Path

import pandas as pd
import torch
from jiwer import wer, mer, process_words
from tqdm import tqdm

from src.config import LOCAL_L2ARCTIC_DIR, MODELS_DIR, NLTK_DATA_PATH
from src.utils.load_l2arctic import load_test_utterances

# --- NLTK / G2P setup ---
import os
import nltk
nltk.data.path.insert(0, NLTK_DATA_PATH)
os.environ["NLTK_DATA"] = NLTK_DATA_PATH

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
    def __init__(self, base_model_path, adapter_path, device="cuda", batch_size=8):
        from models.whisfusion.src.evaluation.evaluate_whisfusion import WhisfusionBenchmark

        self.device = device
        self.batch_size = batch_size

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
            mask_ratio_schedule=[1.0, 0.9, 0.85, 0.8],
        )

        best = max(candidates, key=lambda x: x["avg_confidence"])
        return best["text"].strip()

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

# ---------------------------------------------------------------------------
# Load encoded features (for eval)
# ---------------------------------------------------------------------------
def build_pt_dataset(utterances, processed_root="data/processed/test", split="scripted"):
    pt_root = Path(processed_root) / split

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

def evaluate(dataset, model: WhisfusionWrapper):
    rows = []

    pt_paths = [d["pt_path"] for d in dataset]
    predictions = model.transcribe_from_hidden_states_batch(pt_paths)

    assert len(predictions) == len(dataset)


    for d, pred in zip(dataset, predictions):
        ref = norm(d["text"])
        pred_n = norm(pred)

        word_measures = process_words(ref, pred_n) if ref else None

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--split", default="scripted", choices=["scripted", "spontaneous"])
    parser.add_argument("--output_dir", default="results/model_perf_comparison")
    parser.add_argument("--base_model_path", default="models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors")
    parser.add_argument("--model", default="whisfusion")

    args = parser.parse_args()

    print(f"Device: {device}")

    adapter_path = f"{MODELS_DIR}/{args.model}/{args.model}_stage2_decoder.pt"
    # check if eval files already exist
    output_file = f"{args.model}_{args.split}_predictions.csv"
    output_path = Path(args.output_dir) / output_file
    if output_path.exists():
        print(f"  [skip] {output_path} already exists — delete to re-run")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # load data
    utterances = load_test_utterances(local_root=args.data_root, split=args.split)
    print(f"Loaded {len(utterances)} utterances")
    dataset = build_pt_dataset(utterances, split=args.split)
    print(f"Using {len(dataset)} utterances with cached features")

    # load model
    model = WhisfusionWrapper(
        base_model_path=args.base_model_path,
        adapter_path=adapter_path,
        device=device
    )

    # run eval
    df = evaluate(dataset, model)

    # save
    df.to_csv(output_path, index=False)

    # metrics
    refs = df["reference_norm"].fillna("").tolist()
    hyps = df["prediction_norm"].fillna("").tolist()

    corpus_measures = process_words(refs, hyps)
    per = df["utt_per"].dropna().mean()

    print(f"\nWER: {corpus_measures.wer:.3f}")
    print(f"MER: {corpus_measures.mer:.3f}")
    print(f"PER: {per:.3f}")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
