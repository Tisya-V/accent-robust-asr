#!/usr/bin/env python3
"""
Precompute and cache DTW alignment paths for bridge training pairs.

For each (L2, native) pair in the mapping JSONs, loads the speech frames from
their encoder state files, computes the DTW warping path, and saves it as a
.npy file. The mapping JSONs are updated in-place with a `dtw_path` field
pointing to the cached file.

Run on RDS (CPU-only):
  python -m src.experiments.exp2_latent_diffusion_bridge.precompute_dtw \\
      --cache_dir data/bridge_dtw_cache \\
      --workers 8

Requires: TRAIN_DATA_DIR and DEV_DATA_DIR env vars (source scripts/env.sh).
"""
import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from dtaidistance import dtw_ndim
from tqdm import tqdm

from src.utils.bridge_utils import get_split_data_dir


def _find_file(rel_path: str) -> Path | None:
    for split in ["train", "dev"]:
        p = get_split_data_dir(split) / rel_path
        if p.exists():
            return p
    return None


def _load_speech_frames(rel_path: str, speech_end: int) -> np.ndarray | None:
    path = _find_file(rel_path)
    if path is None:
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hs = ckpt["hidden_states"]
        arr = hs.to(torch.float32).numpy() if isinstance(hs, torch.Tensor) else np.array(hs, dtype=np.float32)
        return arr[:speech_end]
    except Exception:
        return None


def _compute_group(args: tuple) -> list[tuple[str, str | None]]:
    """Compute DTW paths for all nat pairs sharing the same L2 utterance.

    Loads the L2 encoder state once, then iterates over nat pairs.
    Returns list of (cache_path_str, error_or_None) per pair.
    """
    l2_rel, l2_end, nat_items = args
    # nat_items: list of (nat_rel, nat_end, cache_path_str)

    results = []

    # Skip items already cached
    pending = [(nr, ne, cp) for nr, ne, cp in nat_items if not Path(cp).exists()]
    for _, _, cp in nat_items:
        if Path(cp).exists():
            results.append((cp, None))

    if not pending:
        return results

    # Load L2 speech frames once for all pending nat pairs
    l2_frames = _load_speech_frames(l2_rel, l2_end)
    if l2_frames is None or len(l2_frames) < 2:
        err = f"missing or too-short L2 frames ({len(l2_frames) if l2_frames is not None else 0})"
        return results + [(cp, err) for _, _, cp in pending]

    for nat_rel, nat_end, cache_path_str in pending:
        cache_path = Path(cache_path_str)
        nat_frames = _load_speech_frames(nat_rel, nat_end)
        if nat_frames is None or len(nat_frames) < 2:
            results.append((cache_path_str, "missing or too-short nat frames"))
            continue
        try:
            # path_arr[:, 0] = nat indices, path_arr[:, 1] = l2 indices (matches run_steering convention)
            path = np.array(dtw_ndim.warping_path(nat_frames, l2_frames), dtype=np.int16)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, path)
            results.append((cache_path_str, None))
        except Exception as e:
            results.append((cache_path_str, str(e)))

    return results


def precompute(mapping_paths: list[Path], cache_dir: Path, workers: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build groups: l2_rel → list of (nat_rel, nat_end, cache_path_str)
    # Also track pair → cache_path for writing back to JSON
    groups: dict[tuple, list] = defaultdict(list)  # (l2_rel, l2_end) → nat items
    pair_to_cache: dict[tuple, Path] = {}

    for mapping_path in mapping_paths:
        with open(mapping_path) as f:
            pairs = json.load(f)
        for pair in pairs:
            if "l2_speech_end_frame" not in pair or "nat_speech_end_frame" not in pair:
                print(f"[warn] {mapping_path.name}: pair missing speech end fields — "
                      f"re-run prepare_data.py first.")
                continue
            l2_rel  = pair["l2_encoder_state_path"]
            nat_rel = pair["nat_encoder_state_path"]
            l2_end  = pair["l2_speech_end_frame"]
            nat_end = pair["nat_speech_end_frame"]
            fname   = f"{pair['l2_speaker']}_{pair['nat_speaker']}_{pair['prompt_id']}.npy"
            cache_p = cache_dir / fname
            groups[(l2_rel, l2_end)].append((nat_rel, nat_end, str(cache_p)))
            pair_to_cache[(l2_rel, nat_rel)] = cache_p

    group_items = [(l2_rel, l2_end, nat_items) for (l2_rel, l2_end), nat_items in groups.items()]
    total_pairs = sum(len(nat_items) for _, _, nat_items in group_items)
    already     = sum(1 for _, _, nat_items in group_items
                      for _, _, cp in nat_items if Path(cp).exists())
    print(f"[precompute_dtw] {total_pairs} pairs in {len(group_items)} L2-groups  "
          f"({already} already cached, {total_pairs - already} to compute)")

    if total_pairs - already > 0:
        errors = 0
        # chunksize=1: each group is ~500ms of work so IPC overhead is negligible,
        # and we get live tqdm updates rather than waiting for a large chunk to finish
        with Pool(processes=workers) as pool, tqdm(total=total_pairs, unit="pair") as pbar:
            for group_results in pool.imap_unordered(_compute_group, group_items):
                for _, err in group_results:
                    if err:
                        errors += 1
                pbar.update(len(group_results))
        print(f"[precompute_dtw] Done. {errors} errors.")

    # Write dtw_path field back into each mapping JSON
    for mapping_path in mapping_paths:
        with open(mapping_path) as f:
            pairs = json.load(f)
        updated = 0
        for pair in pairs:
            key = (pair.get("l2_encoder_state_path"), pair.get("nat_encoder_state_path"))
            if key in pair_to_cache and pair_to_cache[key].exists():
                pair["dtw_path"] = str(pair_to_cache[key])
                updated += 1
        with open(mapping_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"[precompute_dtw] {mapping_path.name}: {updated}/{len(pairs)} pairs written with dtw_path")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_train", default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json")
    parser.add_argument("--mapping_dev",   default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json")
    parser.add_argument("--cache_dir",     default="data/bridge_dtw_cache")
    parser.add_argument("--workers",       type=int, default=8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))

    mapping_paths = [project_root / args.mapping_train, project_root / args.mapping_dev]
    missing = [p for p in mapping_paths if not p.exists()]
    if missing:
        sys.exit(f"[precompute_dtw] Missing mapping files: {missing}\nRun prepare_data.py first.")

    precompute(mapping_paths, project_root / args.cache_dir, workers=args.workers)


if __name__ == "__main__":
    main()
