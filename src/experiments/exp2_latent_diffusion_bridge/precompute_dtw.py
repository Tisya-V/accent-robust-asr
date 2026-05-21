#!/usr/bin/env python3
"""
Precompute and cache DTW alignment paths for bridge training pairs.

All paths are stored in a single dict (dtw_paths.pkl) keyed by
(l2_encoder_state_path, nat_encoder_state_path) — loaded entirely into RAM
at BridgeDataset init for O(1) lookup with no per-item NFS opens.

Run on RDS (CPU-only):
  python -m src.experiments.exp2_latent_diffusion_bridge.precompute_dtw \\
      --cache_dir data/bridge_dtw_cache \\
      --workers 8

Requires: TRAIN_DATA_DIR and DEV_DATA_DIR env vars (source scripts/env.sh).
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from dtaidistance import dtw_ndim
from tqdm import tqdm

from src.utils.bridge_utils import get_split_data_dir


def _load_speech_frames(rel_path: str, speech_end: int) -> np.ndarray | None:
    for split in ["train", "dev"]:
        p = get_split_data_dir(split) / rel_path
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                hs = ckpt["hidden_states"]
                arr = hs.to(torch.float32).numpy() if isinstance(hs, torch.Tensor) \
                      else np.array(hs, dtype=np.float32)
                return arr[:speech_end]
            except Exception:
                return None
    return None


def _compute_group(args: tuple) -> list[tuple]:
    """Compute DTW paths for all nat pairs sharing the same L2 utterance.

    Loads L2 speech frames once, iterates over nat pairs.
    Returns list of ((l2_rel, nat_rel), path_arr_or_None, error_or_None).
    """
    l2_rel, l2_end, nat_items = args
    # nat_items: list of (nat_rel, nat_end)

    l2_frames = _load_speech_frames(l2_rel, l2_end)
    if l2_frames is None or len(l2_frames) < 2:
        err = f"missing/short L2 ({len(l2_frames) if l2_frames is not None else 0} frames)"
        return [((l2_rel, nat_rel), None, err) for nat_rel, _ in nat_items]

    results = []
    for nat_rel, nat_end in nat_items:
        key = (l2_rel, nat_rel)
        nat_frames = _load_speech_frames(nat_rel, nat_end)
        if nat_frames is None or len(nat_frames) < 2:
            results.append((key, None, "missing/short nat frames"))
            continue
        try:
            # path[:, 0] = nat indices, path[:, 1] = l2 indices
            path = np.array(dtw_ndim.warping_path(nat_frames, l2_frames), dtype=np.int16)
            results.append((key, path, None))
        except Exception as e:
            results.append((key, None, str(e)))

    return results


def precompute(mapping_paths: list[Path], cache_dir: Path, workers: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cache_dir / "dtw_paths.pkl"

    # Load existing cache to support resumption
    existing: dict = {}
    if pkl_path.exists():
        print(f"[precompute_dtw] Loading existing cache from {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)
        print(f"[precompute_dtw] {len(existing)} paths already cached.")

    # Build groups: (l2_rel, l2_end) → [(nat_rel, nat_end), ...]
    groups: dict[tuple, list] = defaultdict(list)
    all_keys: set = set()

    for mapping_path in mapping_paths:
        with open(mapping_path) as f:
            pairs = json.load(f)
        for pair in pairs:
            if "l2_speech_end_frame" not in pair or "nat_speech_end_frame" not in pair:
                print(f"[warn] {mapping_path.name}: missing speech end fields — "
                      f"re-run prepare_data.py first.")
                continue
            l2_rel  = pair["l2_encoder_state_path"]
            nat_rel = pair["nat_encoder_state_path"]
            key     = (l2_rel, nat_rel)
            all_keys.add(key)
            if key not in existing:
                groups[(l2_rel, pair["l2_speech_end_frame"])].append(
                    (nat_rel, pair["nat_speech_end_frame"])
                )

    pending_groups = [(l2_rel, l2_end, nat_items)
                      for (l2_rel, l2_end), nat_items in groups.items()]
    pending_pairs  = sum(len(g[2]) for g in pending_groups)

    print(f"[precompute_dtw] {len(all_keys)} total pairs — "
          f"{len(existing)} cached, {pending_pairs} to compute")

    if pending_pairs > 0:
        errors  = 0
        new_paths: dict = {}
        with Pool(processes=workers) as pool, \
             tqdm(total=pending_pairs, unit="pair") as pbar:
            for group_results in pool.imap_unordered(_compute_group, pending_groups):
                for key, path, err in group_results:
                    if err:
                        errors += 1
                        print(f"\n  [warn] {key[0]} / {key[1]}: {err}")
                    else:
                        new_paths[key] = path
                pbar.update(len(group_results))

        print(f"[precompute_dtw] Computed {len(new_paths)} paths  ({errors} errors).")
        existing.update(new_paths)

        print(f"[precompute_dtw] Saving {len(existing)} paths to {pkl_path} ...")
        with open(pkl_path, "wb") as f:
            pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[precompute_dtw] Saved ({pkl_path.stat().st_size / 1e6:.1f} MB).")
    else:
        print(f"[precompute_dtw] Nothing to compute.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_train", default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json")
    parser.add_argument("--mapping_dev",   default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json")
    parser.add_argument("--cache_dir",     default="src/experiments/exp2_latent_diffusion_bridge/dtw_cache")
    parser.add_argument("--workers",       type=int, default=8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    mapping_paths = [project_root / args.mapping_train, project_root / args.mapping_dev]
    missing = [p for p in mapping_paths if not p.exists()]
    if missing:
        sys.exit(f"[precompute_dtw] Missing mapping files: {missing}\nRun prepare_data.py first.")

    precompute(mapping_paths, project_root / args.cache_dir, workers=args.workers)


if __name__ == "__main__":
    main()
