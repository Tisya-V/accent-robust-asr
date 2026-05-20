#!/usr/bin/env python3
"""
Latent steering experiment runner.

Usage:
  python src/analysis/run_steering.py --decoder whisper   --method dtw      --out_dir results/e2_steering
  python src/analysis/run_steering.py --decoder whisfusion --method position --out_dir results/e2_steering

Results cached as {out_dir}/{decoder}_{method}_steering.csv — re-run is a no-op if file exists.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import jiwer
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hs = ckpt["hidden_states"]
    return hs.to(torch.float32).cpu().numpy() if isinstance(hs, torch.Tensor) else np.array(hs, dtype=np.float32)


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_ref_lookup(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    prompt_col  = next(c for c in ["utterance_id", "prompt_id", "utt_id"] if c in df.columns)
    speaker_col = next(c for c in ["speaker", "spk"]                      if c in df.columns)
    ref_col     = next(c for c in ["reference_norm", "reference", "text"] if c in df.columns)
    out = {}
    for _, row in df.iterrows():
        val = str(row[ref_col]).strip()
        if pd.notna(val) and val.lower() != "nan":
            out[(str(row[prompt_col]), str(row[speaker_col]))] = val
    return out


def pad_to_1500(steered: np.ndarray, l2_full: np.ndarray, l2_end: int,
                eng_full: np.ndarray, eng_end: int, alpha: float) -> np.ndarray:
    """Pad steered speech to 1500 frames, blending silence from both speakers.
    Exact endpoints: alpha=0 -> l2_full[:1500], alpha=1 -> eng_full[:1500].
    """
    N = len(steered)
    if N >= 1500:
        return steered[:1500].astype(np.float32)
    need = 1500 - N

    def _fit(arr, n):
        return arr[:n] if len(arr) >= n else np.vstack([arr, np.tile(arr[-1:], (n - len(arr), 1))])

    sil = (1 - alpha) * _fit(l2_full[l2_end:], need) + alpha * _fit(eng_full[eng_end:], need)
    return np.vstack([steered, sil]).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_path(info: dict) -> str:
    """Resolve a relative path entry using {SPLIT}_DATA_DIR env vars."""
    rel  = info["path"]
    if Path(rel).is_absolute():
        return rel  # already absolute (old-format mapping)
    split    = info["split"]
    env_var  = f"{split.upper()}_DATA_DIR"
    base     = os.environ.get(env_var)
    if not base:
        sys.exit(f"[run_steering] {env_var} is not set — source your env script first.")
    return str(Path(base) / rel)


def load_data(mapping_cache: Path, ref_csv: Path, num_per_l1: int, seed: int) -> tuple[dict, dict]:
    print("[run_steering] Loading data...")
    with open(mapping_cache) as f:
        mapping = json.load(f)

    # Resolve relative paths to full paths for this environment
    for l1d in mapping.values():
        for info in l1d.values():
            info["path"] = _resolve_path(info)
    ref_data = build_ref_lookup(str(ref_csv))
    ref_data = {k: v for k, v in ref_data.items() if pd.notna(v) and v.lower() != "nan"}

    mapping = {
        pid: {l1: info for l1, info in l1d.items() if (pid, info["speaker"]) in ref_data}
        for pid, l1d in mapping.items()
    }
    mapping = {pid: d for pid, d in mapping.items() if "English" in d and len(d) > 1}
    print(f"[run_steering] Prompts after filtering: {len(mapping)}")

    np.random.seed(seed)
    if num_per_l1 > 0:
        selected = set()
        for l1 in sorted({l1 for l1d in mapping.values() for l1 in l1d if l1 != "English"}):
            candidates = [pid for pid, l1d in mapping.items() if l1 in l1d]
            chosen = np.random.choice(candidates, size=min(num_per_l1, len(candidates)), replace=False)
            selected.update(chosen)
            print(f"  {l1}: {len(chosen)} prompts")
        mapping = {pid: mapping[pid] for pid in selected}
        print(f"[run_steering] Stratified subsample: {len(mapping)} prompts total")

    return mapping, ref_data


# ---------------------------------------------------------------------------
# Decoder loading
# ---------------------------------------------------------------------------

def load_decoder(decoder: str, project_root: Path,
                 wf_base_model: str, wf_adapter: str, device: str) -> Callable:
    print(f"[run_steering] Loading {decoder} decoder...")

    if decoder == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.modeling_outputs import BaseModelOutput

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval().to(device)
        proc  = WhisperProcessor.from_pretrained("openai/whisper-small")

        @torch.inference_mode()
        def decode(hidden_state_np: np.ndarray) -> str:
            enc = BaseModelOutput(
                last_hidden_state=torch.from_numpy(hidden_state_np).unsqueeze(0).to(device)
            )
            ids = model.generate(encoder_outputs=enc, language="en", task="transcribe", temperature=0.0)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    else:  # whisfusion
        from src.training.evaluation.eval_whisfusion import WhisfusionWrapper

        wrapper = WhisfusionWrapper(
            base_model_path=str(project_root / wf_base_model),
            adapter_path=str(project_root / wf_adapter),
            device=device,
        )

        @torch.inference_mode()
        def decode(hidden_state_np: np.ndarray) -> str:
            return wrapper.decode(torch.from_numpy(hidden_state_np).to(device))

    print(f"[run_steering] Decoder ready.")
    return decode


# ---------------------------------------------------------------------------
# Steering methods
# ---------------------------------------------------------------------------

def run_position(mapping: dict, ref_data: dict, decode: Callable) -> list[dict]:
    print("[run_steering] Running position-based steering...")
    n_l2 = lambda l1d: len([l for l in l1d if l != "English"])
    total = sum(n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            try:
                eng_full = load_checkpoint(l1d["English"]["path"])
            except Exception:
                pbar.update(n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                ref = ref_data.get((prompt_id, info["speaker"]))
                if ref is None:
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full = load_checkpoint(info["path"])
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                for alpha in ALPHA_VALUES:
                    try:
                        steered = ((1 - alpha) * l2_full + alpha * eng_full).astype(np.float32)
                        pred    = decode(steered)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm(ref), norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_dtw(mapping: dict, ref_data: dict, decode: Callable) -> list[dict]:
    from dtaidistance import dtw_ndim

    print("[run_steering] Running DTW-alpha-timeline steering...")
    n_l2 = lambda l1d: len([l for l in l1d if l != "English"])
    total = sum(n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            eng_end  = eng_info.get("speech_end_frame")
            if not eng_end or not Path(eng_info["path"]).exists():
                pbar.update(n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full  = load_checkpoint(eng_info["path"])
                eng_state = eng_full[:eng_end]
            except Exception:
                pbar.update(n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                l2_end = info.get("speech_end_frame")
                ref    = ref_data.get((prompt_id, info["speaker"]))
                if not l2_end or ref is None or not Path(info["path"]).exists():
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full  = load_checkpoint(info["path"])
                    l2_state = l2_full[:l2_end]
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                path_arr = np.array(dtw_ndim.warping_path(eng_state, l2_state))
                T_eng, T_l2 = len(eng_state), len(l2_state)
                i_norm = path_arr[:, 0] / max(T_eng - 1, 1)
                j_norm = path_arr[:, 1] / max(T_l2  - 1, 1)

                for alpha in ALPHA_VALUES:
                    try:
                        N     = max(1, round((1 - alpha) * T_l2 + alpha * T_eng))
                        t_k   = (1 - alpha) * j_norm + alpha * i_norm
                        out_t = np.linspace(0.0, 1.0, N)
                        idx_r = np.clip(np.searchsorted(t_k, out_t), 0, len(t_k) - 1)
                        idx_l = np.clip(idx_r - 1, 0, len(t_k) - 1)
                        k_idx = np.where(
                            np.abs(t_k[idx_l] - out_t) <= np.abs(t_k[idx_r] - out_t),
                            idx_l, idx_r)
                        steered = ((1 - alpha) * l2_state[path_arr[k_idx, 1]]
                                   + alpha     * eng_state[path_arr[k_idx, 0]]).astype(np.float32)
                        padded  = pad_to_1500(steered, l2_full, l2_end, eng_full, eng_end, alpha)
                        pred    = decode(padded)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm(ref), norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder",    required=True, choices=["whisper", "whisfusion"])
    parser.add_argument("--method",     required=True, choices=["position", "dtw"])
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--num_per_l1", type=int, default=100,
                        help="Utterances per L1 (0 = full dataset)")
    parser.add_argument("--mapping_cache",
                        default="src/analysis/cache/utterance_mapping.json")
    parser.add_argument("--ref_csv",
                        default="results/model_perf_comparison/whisfusion_finetuned_predictions.csv")
    parser.add_argument("--wf_base_model",
                        default="models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors")
    parser.add_argument("--wf_adapter",
                        default="models/whisfusion_finetuned/stage2_decoder/whisfusion_stage2_decoder.pt")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.config import RANDOM_SEED

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{args.decoder}_{args.method}_steering.csv"

    if cache_path.exists():
        print(f"[run_steering] Already cached at {cache_path} — nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_steering] decoder={args.decoder}  method={args.method}  device={device}")

    mapping, ref_data = load_data(
        project_root / args.mapping_cache,
        project_root / args.ref_csv,
        args.num_per_l1,
        RANDOM_SEED,
    )
    decode = load_decoder(args.decoder, project_root, args.wf_base_model, args.wf_adapter, device)

    rows = run_position(mapping, ref_data, decode) if args.method == "position" \
        else run_dtw(mapping, ref_data, decode)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"[run_steering] Saved {len(df)} rows to {cache_path}")
    print(df.groupby("alpha")["wer"].agg(["mean", "std", "min", "max"]).round(4))


if __name__ == "__main__":
    main()
