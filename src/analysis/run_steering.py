#!/usr/bin/env python3
"""
Latent steering experiment runner.

Methods:
  position     — frame-by-frame blend of full 1500-frame sequences; no tail option
  position_fixed — positional blend up to max(T_l2, T_eng); L2 tail
  position_nt  — positional blend up to N(alpha) = round((1-a)*T_l2 + a*T_eng); L2 silence tail
  dtw          — DTW-aligned speech frames + configurable tail (--tail l2|english|interpolate)
  dtw_fixed    — fixed DTW mapping on L2 timeline; blend each L2 frame towards its DTW-matched
                 native frame; output always T_l2 frames + L2 silence tail; no N(alpha) morphing
  full_dtw     — DTW aligned across all 1500 frames; no tail handling needed

Output files:
  {out_dir}/{decoder}_position_steering.csv
  {out_dir}/{decoder}_position_fixed_steering.csv
  {out_dir}/{decoder}_position_nt_steering.csv
  {out_dir}/{decoder}_dtw_{tail}_steering.csv
  {out_dir}/{decoder}_dtw_fixed_steering.csv
  {out_dir}/{decoder}_full_dtw_steering.csv

Re-running is a no-op if the output file already exists.
"""
import argparse
import json
import os
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

def _n_l2(l1d: dict) -> int:
    return sum(1 for l1 in l1d if l1 != "English")


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


def _extend_sil(sil: np.ndarray, need: int) -> np.ndarray:
    """Extend a silence slice to exactly `need` frames by tiling the last frame."""
    if len(sil) == 0:
        # Silence region is empty (speech fills entire 1500 frames); pad with zeros
        return np.zeros((need, sil.shape[-1]), dtype=np.float32)
    if len(sil) >= need:
        return sil[:need]
    return np.vstack([sil, np.tile(sil[-1:], (need - len(sil), 1))])


def pad_to_1500(steered: np.ndarray, l2_full: np.ndarray, l2_end: int,
                eng_full: np.ndarray, eng_end: int, alpha: float,
                tail: str = "l2") -> np.ndarray:
    """Append silence frames to bring steered speech content to 1500 frames.

    tail='l2'          use L2 silence — realistic (only signal at bridge inference)
    tail='english'     use English silence — oracle upper bound
    tail='interpolate' alpha-blend L2/English silence — consistent at both endpoints

    alpha=0 always returns l2_full[:1500] regardless of tail strategy.
    """
    if alpha == 0.0:
        return l2_full[:1500].astype(np.float32)

    N = len(steered)
    if N >= 1500:
        return steered[:1500].astype(np.float32)
    need = 1500 - N

    if tail == "english":
        padding = _extend_sil(eng_full[eng_end:], need)
    elif tail == "interpolate":
        l2_pad  = _extend_sil(l2_full[l2_end:],  need)
        eng_pad = _extend_sil(eng_full[eng_end:], need)
        padding = ((1 - alpha) * l2_pad + alpha * eng_pad).astype(np.float32)
    else:  # l2
        padding = _extend_sil(l2_full[l2_end:], need)

    return np.vstack([steered, padding]).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_path(info: dict) -> str:
    rel = info["path"]
    if Path(rel).is_absolute():
        return rel
    env_var = f"{info['split'].upper()}_DATA_DIR"
    base    = os.environ.get(env_var)
    if not base:
        sys.exit(f"[run_steering] {env_var} is not set — source your env script first.")
    return str(Path(base) / rel)


def load_data(mapping_cache: Path, ref_csv: Path, num_prompts: int, seed: int) -> tuple[dict, dict]:
    print("[run_steering] Loading data...")
    with open(mapping_cache) as f:
        mapping = json.load(f)

    for l1d in mapping.values():
        for info in l1d.values():
            info["path"] = _resolve_path(info)

    ref_data = build_ref_lookup(str(ref_csv))

    mapping = {
        pid: {l1: info for l1, info in l1d.items() if (pid, info["speaker"]) in ref_data}
        for pid, l1d in mapping.items()
    }
    mapping = {pid: d for pid, d in mapping.items() if "English" in d and len(d) > 1}
    print(f"[run_steering] Prompts after filtering: {len(mapping)}")

    np.random.seed(seed)
    if num_prompts > 0:
        all_l1s = {l1 for l1d in mapping.values() for l1 in l1d if l1 != "English"}
        full    = sorted(pid for pid, l1d in mapping.items() if all_l1s <= l1d.keys())
        partial = sorted(pid for pid in mapping if pid not in set(full))
        pool    = full + partial
        chosen  = np.random.choice(pool, size=min(num_prompts, len(pool)), replace=False)
        mapping = {pid: mapping[pid] for pid in chosen}
        n_full  = sum(1 for pid in chosen if pid in set(full))
        print(f"[run_steering] Sampled {len(mapping)} prompts  ({n_full} with full L1 coverage)")

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

    print("[run_steering] Decoder ready.")
    return decode


# ---------------------------------------------------------------------------
# Steering methods
# ---------------------------------------------------------------------------

def run_position_fixed(mapping: dict, ref_data: dict, decode: Callable) -> list[dict]:
    """Position blend of speech frames [0:T_max] only; L2 padding for tail.

    T_max = max(T_nat, T_l2) — covers the full speech region of both speakers.
    Tail [T_max:] is always L2 padding (no interpolation).
    Compared to run_position (full 1500-frame blend), this isolates whether fixing
    the tail to on-manifold L2 content improves WER. Requires speech_end_frame in mapping.
    """
    print("[run_steering] Running fixed-tail position-based steering...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            eng_end  = eng_info.get("speech_end_frame")
            if not eng_end or not Path(eng_info["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full = load_checkpoint(eng_info["path"])
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                l2_end = info.get("speech_end_frame")
                ref    = ref_data.get((prompt_id, info["speaker"]))
                if not l2_end or ref is None or not Path(info["path"]).exists():
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full = load_checkpoint(info["path"])
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                t_max    = max(l2_end, eng_end)
                norm_ref = norm(ref)
                for alpha in ALPHA_VALUES:
                    try:
                        speech = ((1 - alpha) * l2_full[:t_max]
                                  + alpha * eng_full[:t_max]).astype(np.float32)
                        l2_sil = _extend_sil(l2_full[l2_end:], 1500 - t_max)
                        padded = np.vstack([speech, l2_sil])
                        pred   = decode(padded)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_position(mapping: dict, ref_data: dict, decode: Callable) -> list[dict]:
    """Frame-by-frame blend of full 1500-frame sequences. Tail is implicitly interpolated."""
    print("[run_steering] Running position-based steering...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            if not Path(l1d["English"]["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full = load_checkpoint(l1d["English"]["path"])
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                ref = ref_data.get((prompt_id, info["speaker"]))
                if ref is None or not Path(info["path"]).exists():
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full = load_checkpoint(info["path"])
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                norm_ref = norm(ref)
                for alpha in ALPHA_VALUES:
                    try:
                        steered = ((1 - alpha) * l2_full + alpha * eng_full).astype(np.float32)
                        pred    = decode(steered)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_position_nt(mapping: dict, ref_data: dict, decode: Callable) -> list[dict]:
    """Position blend with N(alpha)-varying active region, matching DTW's masking convention.

    N(alpha) = round((1-alpha)*T_l2 + alpha*T_eng) frames blended positionally.
    Tail [N:] is always L2 silence (_extend_sil(l2_full[T_l2:], ...)), same as DTW's
    _apply_mask — isolates whether N(t) masking helps independent of DTW alignment.
    """
    print("[run_steering] Running position N(t) steering...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            eng_end  = eng_info.get("speech_end_frame")
            if not eng_end or not Path(eng_info["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full = load_checkpoint(eng_info["path"])
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                l2_end = info.get("speech_end_frame")
                ref    = ref_data.get((prompt_id, info["speaker"]))
                if not l2_end or ref is None or not Path(info["path"]).exists():
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full = load_checkpoint(info["path"])
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                norm_ref = norm(ref)
                for alpha in ALPHA_VALUES:
                    try:
                        N      = max(1, round((1 - alpha) * l2_end + alpha * eng_end))
                        speech = ((1 - alpha) * l2_full[:N]
                                  + alpha * eng_full[:N]).astype(np.float32)
                        need   = 1500 - N
                        if need > 0:
                            l2_sil = _extend_sil(l2_full[l2_end:], need)
                            padded = np.vstack([speech, l2_sil])
                        else:
                            padded = speech[:1500]
                        pred   = decode(padded)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_dtw(mapping: dict, ref_data: dict, decode: Callable,
            tail: str = "l2", window: int | None = None) -> list[dict]:
    """DTW-aligned speech frames with configurable tail padding to 1500 frames."""
    from dtaidistance import dtw_ndim

    print(f"[run_steering] Running DTW-alpha-timeline steering (tail={tail}, window={window})...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            eng_end  = eng_info.get("speech_end_frame")
            if not eng_end or not Path(eng_info["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full  = load_checkpoint(eng_info["path"])
                eng_state = eng_full[:eng_end]
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

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

                path_arr = np.array(dtw_ndim.warping_path(eng_state, l2_state, window=window))
                T_eng, T_l2 = len(eng_state), len(l2_state)
                i_norm = path_arr[:, 0] / max(T_eng - 1, 1)
                j_norm = path_arr[:, 1] / max(T_l2  - 1, 1)
                norm_ref = norm(ref)

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
                        padded  = pad_to_1500(steered, l2_full, l2_end, eng_full, eng_end, alpha, tail=tail)
                        pred    = decode(padded)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_dtw_fixed(mapping: dict, ref_data: dict, decode: Callable,
                  window: int | None = None) -> list[dict]:
    """DTW-fixed steering: freeze DTW mapping on L2 timeline, blend towards native in-place.

    For each L2 speech frame k, find its DTW-matched native frame nat_idx[k] by projecting
    the warping path onto the L2 frame grid (nearest-neighbour on j_norm). This is computed
    once per pair. Then for each alpha:

        steered[k] = (1-alpha)*z_l2[k] + alpha*z_nat[nat_idx[k]]

    Output is always T_l2 speech frames + original L2 silence padding to 1500.
    No N(alpha) morphing — tests whether the fixed-L2-timeline approach (used by the
    planned dtw_fixed bridge variant) retains coherent blend properties.
    """
    from dtaidistance import dtw_ndim

    print(f"[run_steering] Running fixed-timeline DTW steering (window={window})...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            eng_end  = eng_info.get("speech_end_frame")
            if not eng_end or not Path(eng_info["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full  = load_checkpoint(eng_info["path"])
                eng_state = eng_full[:eng_end]
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

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

                path_arr = np.array(dtw_ndim.warping_path(eng_state, l2_state, window=window))
                T_l2 = len(l2_state)

                # Project path onto L2 grid: for each L2 frame k, find nearest path point
                j_norm  = path_arr[:, 1].astype(np.float32) / max(T_l2 - 1, 1)
                out_t   = np.linspace(0.0, 1.0, T_l2, dtype=np.float32)
                idx_r   = np.clip(np.searchsorted(j_norm, out_t), 0, len(j_norm) - 1)
                idx_l   = np.clip(idx_r - 1, 0, len(j_norm) - 1)
                k_idx   = np.where(
                    np.abs(j_norm[idx_l] - out_t) <= np.abs(j_norm[idx_r] - out_t),
                    idx_l, idx_r)
                nat_idx = path_arr[k_idx, 0]  # fixed: nat frame for each L2 frame k

                norm_ref = norm(ref)
                for alpha in ALPHA_VALUES:
                    try:
                        speech = ((1 - alpha) * l2_state
                                  + alpha * eng_state[nat_idx]).astype(np.float32)
                        padded = pad_to_1500(speech, l2_full, l2_end,
                                             eng_full, eng_end, alpha, tail="l2")
                        pred   = decode(padded)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


def run_dtw_full(mapping: dict, ref_data: dict, decode: Callable,
                 window: int | None = None) -> list[dict]:
    """DTW across all 1500 frames — no speech-end detection, no tail handling.
    Output is always 1500 frames. More expensive than speech-only DTW; use --window.
    """
    from dtaidistance import dtw_ndim

    print(f"[run_steering] Running full-sequence DTW steering (window={window})...")
    total = sum(_n_l2(l1d) * len(ALPHA_VALUES) for l1d in mapping.values())
    rows  = []
    out_t = np.linspace(0.0, 1.0, 1500)  # fixed output grid, same for every alpha

    with tqdm(total=total, unit="decode") as pbar:
        for prompt_id, l1d in mapping.items():
            eng_info = l1d["English"]
            if not Path(eng_info["path"]).exists():
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue
            try:
                eng_full = load_checkpoint(eng_info["path"])[:1500]
            except Exception:
                pbar.update(_n_l2(l1d) * len(ALPHA_VALUES)); continue

            for l1, info in l1d.items():
                if l1 == "English":
                    continue
                ref = ref_data.get((prompt_id, info["speaker"]))
                if ref is None or not Path(info["path"]).exists():
                    pbar.update(len(ALPHA_VALUES)); continue
                try:
                    l2_full = load_checkpoint(info["path"])[:1500]
                except Exception:
                    pbar.update(len(ALPHA_VALUES)); continue

                # DTW computed once per pair and reused across all alphas
                path_arr = np.array(dtw_ndim.warping_path(eng_full, l2_full, window=window))
                i_norm   = path_arr[:, 0] / 1499.0
                j_norm   = path_arr[:, 1] / 1499.0
                norm_ref = norm(ref)

                for alpha in ALPHA_VALUES:
                    try:
                        t_k   = (1 - alpha) * j_norm + alpha * i_norm
                        idx_r = np.clip(np.searchsorted(t_k, out_t), 0, len(t_k) - 1)
                        idx_l = np.clip(idx_r - 1, 0, len(t_k) - 1)
                        k_idx = np.where(
                            np.abs(t_k[idx_l] - out_t) <= np.abs(t_k[idx_r] - out_t),
                            idx_l, idx_r)
                        steered = ((1 - alpha) * l2_full[path_arr[k_idx, 1]]
                                   + alpha     * eng_full[path_arr[k_idx, 0]]).astype(np.float32)
                        pred = decode(steered)
                        rows.append({"prompt_id": prompt_id, "L1": l1,
                                     "speaker": info["speaker"], "alpha": alpha,
                                     "wer": jiwer.wer(norm_ref, norm(pred))})
                    except Exception as e:
                        print(f"  [warn] {prompt_id}/{l1}/alpha={alpha}: {e}")
                    pbar.update(1)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder",      required=True, choices=["whisper", "whisfusion"])
    parser.add_argument("--method",       required=True, choices=["position", "position_fixed", "position_nt", "dtw", "dtw_fixed", "full_dtw"])
    parser.add_argument("--tail",         default="l2",  choices=["l2", "english", "interpolate"],
                        help="Tail/padding strategy for --method dtw only. "
                             "Ignored for position (blends all 1500 frames), "
                             "position_fixed (L2 tail after T_max), and "
                             "full_dtw (DTW across all 1500 frames; no tail needed).")
    parser.add_argument("--window",       type=int, default=None,
                        help="Sakoe-Chiba band for DTW (frames). Strongly recommended for "
                             "full_dtw (e.g. --window 200) to keep runtime practical.")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--num_prompts",  type=int, default=100,
                        help="Prompts to sample (0 = full dataset); prefers full L1 coverage.")
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

    if args.method == "dtw":
        cache_path = out_dir / f"{args.decoder}_dtw_{args.tail}_steering.csv"
    elif args.method == "dtw_fixed":
        cache_path = out_dir / f"{args.decoder}_dtw_fixed_steering.csv"
    elif args.method == "full_dtw":
        cache_path = out_dir / f"{args.decoder}_full_dtw_steering.csv"
    elif args.method == "position_fixed":
        cache_path = out_dir / f"{args.decoder}_position_fixed_steering.csv"
    elif args.method == "position_nt":
        cache_path = out_dir / f"{args.decoder}_position_nt_steering.csv"
    else:
        cache_path = out_dir / f"{args.decoder}_position_steering.csv"

    if cache_path.exists():
        print(f"[run_steering] Already cached at {cache_path} — nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_steering] decoder={args.decoder}  method={args.method}  device={device}")

    mapping, ref_data = load_data(
        project_root / args.mapping_cache,
        project_root / args.ref_csv,
        args.num_prompts,
        RANDOM_SEED,
    )
    decode = load_decoder(args.decoder, project_root, args.wf_base_model, args.wf_adapter, device)

    if args.method == "position":
        rows = run_position(mapping, ref_data, decode)
    elif args.method == "position_fixed":
        rows = run_position_fixed(mapping, ref_data, decode)
    elif args.method == "position_nt":
        rows = run_position_nt(mapping, ref_data, decode)
    elif args.method == "dtw_fixed":
        rows = run_dtw_fixed(mapping, ref_data, decode, window=args.window)
    elif args.method == "full_dtw":
        rows = run_dtw_full(mapping, ref_data, decode, window=args.window)
    else:
        rows = run_dtw(mapping, ref_data, decode, tail=args.tail, window=args.window)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"[run_steering] Saved {len(df)} rows to {cache_path}")
    print(df.groupby("alpha")["wer"].agg(["mean", "std", "min", "max"]).round(4))


if __name__ == "__main__":
    main()
