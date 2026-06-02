#!/usr/bin/env python3
"""
Prepare utterance mappings for E2 Latent Diffusion Bridge — v2 split.

Differences from prepare_data.py:
  - stratified_split_by_prompt_id(): splits 85/15 at the (l2_speaker, prompt_id)
    slot level rather than at the pair level. All native speaker pairings for a
    given L2 utterance land in the same split, so no z_acc appears in both
    train and dev under different native speaker targets.
  - build_dedup_mapping(): for Branch A — selects the single closest native
    speaker (by frame-averaged cosine similarity over speech frames) per slot
    in the train mapping only. Dev is NOT deduplicated.
  - Writes:  mapping_train_v2.json, mapping_dev_v2.json (all conditions)
             mapping_train_dedup.json             (Branch A training)
  - Does NOT overwrite mapping_train.json or mapping_dev.json.

Usage:
  python -m src.experiments.exp2_latent_diffusion_bridge.prepare_data_v2 \
      --output_dir src/experiments/exp2_latent_diffusion_bridge/data

  # To skip the dedup step (test split logic only):
  python -m ... --skip_dedup
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.bridge_utils import get_split_data_dir


# ---------------------------------------------------------------------------
# Step 1 — prompt_id-level split
# ---------------------------------------------------------------------------

def stratified_split_by_prompt_id(
    pairs: List[Dict],
    train_ratio: float = 0.85,
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split 85/15 at the (l2_speaker, prompt_id) slot level, stratified by l1.

    All native speaker pairings for a given slot go to either train or dev,
    never both. This prevents the same z_acc from appearing in both splits
    under different native speaker targets.

    Returns: (train_pairs, dev_pairs)
    """
    slots: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for pair in pairs:
        slot = (pair["l2_speaker"], pair["prompt_id"])
        slots[slot].append(pair)

    slot_keys = list(slots.keys())
    slot_l1s  = [slots[k][0]["l1"] for k in slot_keys]

    l1_counts: Dict[str, int] = defaultdict(int)
    for l1 in slot_l1s:
        l1_counts[l1] += 1

    print(f"\n[stratified_split_by_prompt_id]")
    print(f"  Total slots: {len(slot_keys)}")
    for l1 in sorted(l1_counts):
        print(f"  {l1}: {l1_counts[l1]} slots")

    train_keys, dev_keys = train_test_split(
        slot_keys,
        train_size=train_ratio,
        stratify=slot_l1s,
        random_state=random_seed,
    )
    train_key_set = set(train_keys)

    train_pairs: List[Dict] = []
    dev_pairs:   List[Dict] = []
    for key, slot_pairs in slots.items():
        if key in train_key_set:
            for p in slot_pairs:
                p["bridge_split"] = "train"
            train_pairs.extend(slot_pairs)
        else:
            for p in slot_pairs:
                p["bridge_split"] = "val"
            dev_pairs.extend(slot_pairs)

    print(f"  Train: {len(train_pairs)} pairs across {len(train_keys)} slots "
          f"({100 * len(train_pairs) / len(pairs):.1f}%)")
    print(f"  Dev:   {len(dev_pairs)} pairs across {len(dev_keys)} slots "
          f"({100 * len(dev_pairs) / len(pairs):.1f}%)")
    return train_pairs, dev_pairs


# ---------------------------------------------------------------------------
# Step 2 — Branch A dedup mapping
# ---------------------------------------------------------------------------

def _find_encoder_state(rel_path: str) -> Path | None:
    """Resolve a relative encoder state path by searching train/dev split dirs."""
    for split_name in ["train", "dev"]:
        p = get_split_data_dir(split_name) / rel_path
        if p.exists():
            return p
    return None


def _frame_avg_cos_sim(l2_path: Path, nat_path: Path,
                       l2_end: int, nat_end: int) -> float:
    """Frame-averaged cosine similarity over speech frames only."""
    z_acc = torch.load(l2_path,  map_location="cpu", weights_only=False)["hidden_states"].float()
    z_nat = torch.load(nat_path, map_location="cpu", weights_only=False)["hidden_states"].float()
    T = min(l2_end, nat_end)
    if T <= 0:
        return 0.0
    return F.cosine_similarity(z_acc[:T], z_nat[:T], dim=-1).mean().item()


def build_dedup_mapping(train_pairs: List[Dict], skip_missing: bool = True) -> List[Dict]:
    """
    Branch A: select the single native speaker with the highest frame-averaged
    cosine similarity to z_acc, per (l2_speaker, prompt_id) slot.

    Applied to train_pairs only. Dev is NOT deduplicated — all branches
    validate against the full mapping_dev_v2.json using min-over-natives loss.

    Returns a deduplicated list: one entry per slot.
    """
    slots: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for pair in train_pairs:
        slot = (pair["l2_speaker"], pair["prompt_id"])
        slots[slot].append(pair)

    print(f"\n[build_dedup_mapping] {len(slots)} slots from {len(train_pairs)} pairs")

    # Resolve all encoder state paths in parallel (NFS stat is the bottleneck)
    all_rel = list({p for pair in train_pairs
                    for p in (pair["l2_encoder_state_path"], pair["nat_encoder_state_path"])})
    with ThreadPoolExecutor(max_workers=16) as ex:
        resolved = dict(zip(all_rel, ex.map(_find_encoder_state, all_rel)))

    dedup:   List[Dict] = []
    skipped: int        = 0

    for slot_pairs in tqdm(slots.values(), desc="Selecting closest native speaker"):
        best_pair: Dict | None = None
        best_sim:  float       = -2.0

        for pair in slot_pairs:
            l2_path  = resolved.get(pair["l2_encoder_state_path"])
            nat_path = resolved.get(pair["nat_encoder_state_path"])
            if l2_path is None or nat_path is None:
                if not skip_missing:
                    raise FileNotFoundError(
                        f"Missing encoder state for {pair['l2_encoder_state_path']}"
                    )
                continue
            try:
                sim = _frame_avg_cos_sim(
                    l2_path, nat_path,
                    pair["l2_speech_end_frame"], pair["nat_speech_end_frame"],
                )
            except Exception as e:
                print(f"[WARN] cos_sim failed: {e}")
                continue
            if sim > best_sim:
                best_sim  = sim
                best_pair = pair

        if best_pair is not None:
            dedup.append(best_pair)
        else:
            skipped += 1

    print(f"[build_dedup_mapping] {len(dedup)} entries selected, {skipped} slots skipped")

    nat_dist: Dict[str, int] = defaultdict(int)
    for p in dedup:
        nat_dist[p["nat_speaker"]] += 1
    print("[build_dedup_mapping] Native speaker distribution:")
    for spk in sorted(nat_dist):
        print(f"  {spk}: {nat_dist[spk]} ({100 * nat_dist[spk] / max(len(dedup), 1):.1f}%)")

    return dedup


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_data_v2(
    output_dir:        str = "src/experiments/exp2_latent_diffusion_bridge/data",
    mapping_train_src: str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
    mapping_dev_src:   str = "src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
    skip_dedup:        bool = False,
) -> None:
    """Re-split existing pair mappings by prompt_id slot.

    Loads all pairs from the already-built mapping_train.json + mapping_dev.json,
    applies the slot-level 85/15 split, and writes the v2 files. Does NOT
    require raw audio or L2-ARCTIC metadata to be present locally.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Loading existing mappings]")
    with open(mapping_train_src) as f:
        train_src = json.load(f)
    with open(mapping_dev_src) as f:
        dev_src = json.load(f)
    pairs = train_src + dev_src
    print(f"  {len(train_src)} train + {len(dev_src)} dev = {len(pairs)} total pairs")

    # v2 split: by (l2_speaker, prompt_id) slot, not by pair
    train_pairs, dev_pairs = stratified_split_by_prompt_id(
        pairs, train_ratio=0.85, random_seed=42
    )

    # Verify disjoint slots
    train_slots = {(p["l2_speaker"], p["prompt_id"]) for p in train_pairs}
    dev_slots   = {(p["l2_speaker"], p["prompt_id"]) for p in dev_pairs}
    overlap = train_slots & dev_slots
    if overlap:
        print(f"[ERROR] {len(overlap)} slots appear in both train and dev!")
    else:
        print(f"[OK] Train/dev slots are fully disjoint.")

    train_v2_path = output_dir / "mapping_train_v2.json"
    dev_v2_path   = output_dir / "mapping_dev_v2.json"
    with open(train_v2_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(dev_v2_path, "w") as f:
        json.dump(dev_pairs, f, indent=2)
    print(f"\n[Saved] {train_v2_path} ({len(train_pairs)} pairs)")
    print(f"[Saved] {dev_v2_path}   ({len(dev_pairs)} pairs)")

    # Dedup mapping (train only; requires encoder states to be accessible)
    dedup_pairs: List[Dict] = []
    if not skip_dedup:
        dedup_pairs = build_dedup_mapping(train_pairs)
        dedup_path  = output_dir / "mapping_train_dedup.json"
        with open(dedup_path, "w") as f:
            json.dump(dedup_pairs, f, indent=2)
        print(f"[Saved] {dedup_path} ({len(dedup_pairs)} pairs)")

    print("\n[Summary]")
    print(f"  mapping_train_v2.json:    {len(train_pairs)}")
    print(f"  mapping_dev_v2.json:      {len(dev_pairs)}")
    if not skip_dedup:
        print(f"  mapping_train_dedup.json: {len(dedup_pairs)}")
    print("\n  mapping_train.json and mapping_dev.json are UNCHANGED.")
    print("\n[✅ Data preparation v2 complete]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare v2 utterance mappings for E2 Latent Diffusion Bridge"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data",
        help="Output directory for JSON mappings",
    )
    parser.add_argument(
        "--mapping_train_src", type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json",
        help="Existing mapping_train.json to re-split",
    )
    parser.add_argument(
        "--mapping_dev_src", type=str,
        default="src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json",
        help="Existing mapping_dev.json to re-split",
    )
    parser.add_argument(
        "--skip_dedup", action="store_true",
        help="Skip build_dedup_mapping (test split logic without loading .pt files)",
    )
    args = parser.parse_args()
    prepare_data_v2(
        output_dir=args.output_dir,
        mapping_train_src=args.mapping_train_src,
        mapping_dev_src=args.mapping_dev_src,
        skip_dedup=args.skip_dedup,
    )
