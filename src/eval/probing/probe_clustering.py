"""
probe_clustering.py
Clustering metrics + UMAP visualisations for Whisper encoder representations.

Computes per layer per model:
  - Silhouette score       (phoneme, L1, speaker labels)
  - Davies-Bouldin index   (phoneme, L1, speaker labels)
  - Calinski-Harabasz      (phoneme, L1, speaker labels)
  - Within / between class distance ratio
  - UMAP 2D projections saved as PNG

Usage:
    python probe_clustering.py --models baseline --split scripted
    python probe_clustering.py --models baseline,baseline_lora,ctc_aux --split scripted
    python probe_clustering.py --models ctc_aux --layers 0,4,8,12 \\
        --umap_layers 4,12 --focus_phones TH,DH,V,F,S,Z

Output:
    results/clustering/clustering_{model_key}_{split}.json
    results/clustering/{model_key}/umap_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap as umap_lib
from matplotlib import colormaps
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import L1_GROUPS, LOCAL_L2ARCTIC_DIR, WHISPER_N_ENCODER_LAYERS
from src.eval.probing.probe_utils import (
    build_embedding_dataset,
    records_to_arrays,
    save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry
from src.utils.phonology import ARPABET_VOCAB, PHONE2ID


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------


def clustering_metrics(X: np.ndarray, labels: np.ndarray, label_name: str) -> dict:
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) < len(unique) + 1:
        return {"error": "insufficient_data"}

    if len(X) > 5_000:
        idx = np.random.RandomState(42).choice(len(X), 5_000, replace=False)
        X, labels = X[idx], labels[idx]

    X_scaled = StandardScaler().fit_transform(X)

    try:
        sil = float(silhouette_score(X_scaled, labels, metric="cosine",
                                     sample_size=min(2_000, len(X))))
    except Exception:
        sil = float("nan")
    try:
        db  = float(davies_bouldin_score(X_scaled, labels))
    except Exception:
        db  = float("nan")
    try:
        ch  = float(calinski_harabasz_score(X_scaled, labels))
    except Exception:
        ch  = float("nan")

    try:
        centroids    = {lbl: X_scaled[labels == lbl].mean(0) for lbl in unique}
        within_dists = [
            np.mean(np.linalg.norm(X_scaled[labels == lbl] - centroids[lbl], axis=1))
            for lbl in unique if (labels == lbl).sum() >= 2
        ]
        within_mean  = float(np.mean(within_dists)) if within_dists else float("nan")
        c_arr        = np.stack(list(centroids.values()))
        if len(c_arr) >= 2:
            diffs        = c_arr[:, None] - c_arr[None]
            dists        = np.linalg.norm(diffs, axis=-1)
            mask_tri     = np.triu(np.ones((len(c_arr),) * 2, dtype=bool), k=1)
            between_mean = float(dists[mask_tri].mean())
        else:
            between_mean = float("nan")
        wb_ratio = (float(within_mean / between_mean)
                    if between_mean > 0 else float("nan"))
    except Exception:
        within_mean = between_mean = wb_ratio = float("nan")

    print(f"    {label_name:20s} | sil={sil:.3f}  DB={db:.3f}  W/B={wb_ratio:.3f}")
    return {
        "silhouette":        sil,
        "davies_bouldin":    db,
        "calinski_harabasz": ch,
        "within_mean":       within_mean,
        "between_mean":      between_mean,
        "wb_ratio":          wb_ratio,
        "n_samples":         int(len(X)),
        "n_classes":         int(len(unique)),
    }


# ---------------------------------------------------------------------------
# UMAP plotting
# ---------------------------------------------------------------------------


def make_umap_plot(
    X:            np.ndarray,
    labels:       np.ndarray,
    label_names:  list[str],
    title:        str,
    out_path:     str,
    max_points:   int = 3_000,
    n_neighbors:  int = 15,
    min_dist:     float = 0.1,
) -> None:
    if len(X) > max_points:
        idx    = np.random.RandomState(42).choice(len(X), max_points, replace=False)
        X      = X[idx]
        labels = labels[idx]

    X_scaled  = StandardScaler().fit_transform(X)
    embedding = umap_lib.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="cosine",
        random_state=42, verbose=False,
    ).fit_transform(X_scaled)

    n_cls  = len(label_names)
    cmap   = colormaps["tab20" if n_cls <= 20 else "hsv"].resampled(n_cls)
    colors = [cmap(i / n_cls) for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(label_names):
        mask = labels == i
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[i]], label=name, alpha=0.5, s=10, linewidths=0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2.5, fontsize=7,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    UMAP → {out_path}")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------


def cluster_model(
    model_key:     str,
    model,
    processor,
    utterances:    list,
    layer_indices: list[int],
    umap_layers:   list[int],
    focus_phones:  list[str],
    device:        str,
    output_dir:    str,
    split:         str,
) -> None:
    out_json = Path(output_dir) / f"clustering_{model_key}_{split}.json"
    if out_json.exists():
        print(f"  [skip] {out_json} already exists — delete to re-run")
        return

    print("  Extracting hidden states ...")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records):,} records extracted")

    umap_dir      = Path(output_dir) / model_key
    layer_results = {}

    for layer_idx in layer_indices:
        X, phone_ids, l1_ids, speakers = records_to_arrays(records, layer_idx)
        spk_ints = LabelEncoder().fit_transform(speakers)
        print(f"\\n  Layer {layer_idx} | n={len(X):,}")

        layer_results[str(layer_idx)] = {
            "phoneme": clustering_metrics(X, phone_ids, "phoneme"),
            "l1":      clustering_metrics(X, l1_ids,    "L1/accent"),
            "speaker": clustering_metrics(X, spk_ints,  "speaker"),
        }

        if layer_idx not in umap_layers:
            continue

        base = f"umap_{model_key}_layer{layer_idx}"

        make_umap_plot(X, l1_ids, L1_GROUPS,
            title    = f"{model_key} | Layer {layer_idx} | by L1",
            out_path = str(umap_dir / f"{base}_l1.png"))

        make_umap_plot(X, phone_ids, ARPABET_VOCAB,
            title    = f"{model_key} | Layer {layer_idx} | by phoneme",
            out_path = str(umap_dir / f"{base}_phoneme.png"))

        focus_ids = {PHONE2ID[p] for p in focus_phones if p in PHONE2ID}
        mask_f    = np.isin(phone_ids, list(focus_ids))
        if mask_f.sum() > 20:
            sorted_ids    = sorted(focus_ids)
            id_map        = {old: new for new, old in enumerate(sorted_ids)}
            focus_labels  = [ARPABET_VOCAB[i] for i in sorted_ids]
            pid_remapped  = np.array([id_map[p] for p in phone_ids[mask_f]])

            make_umap_plot(X[mask_f], pid_remapped, focus_labels,
                title    = f"{model_key} | Layer {layer_idx} | focus phones",
                out_path = str(umap_dir / f"{base}_focus_phones.png"))

            make_umap_plot(X[mask_f], l1_ids[mask_f], L1_GROUPS,
                title    = f"{model_key} | Layer {layer_idx} | focus phones by L1",
                out_path = str(umap_dir / f"{base}_focus_phones_l1.png"))

    save_results(layer_results, str(out_json))
    print(f"  Saved → {out_json}")

    print(f"   {'Layer':>6}  {'Phoneme sil':>14}  {'L1 sil':>10}  {'Speaker sil':>14}")
    for li in layer_indices:
        lr  = layer_results[str(li)]
        ph  = lr["phoneme"].get("silhouette", float("nan"))
        l1  = lr["l1"].get("silhouette",      float("nan"))
        spk = lr["speaker"].get("silhouette", float("nan"))
        print(f"  {li:>6}  {ph:>14.3f}  {l1:>10.3f}  {spk:>14.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser(description="Clustering metrics + UMAP for Whisper encoder")
    p.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    p.add_argument("--models",               default="baseline",
                   help="Comma-separated model keys from MODEL_REGISTRY")
    p.add_argument("--split",                default="scripted",
                   choices=["scripted", "spontaneous", "all"])
    p.add_argument("--output_dir",           default="results/clustering")
    p.add_argument("--layers",               default=",".join(str(i) for i in range(WHISPER_N_ENCODER_LAYERS + 1)),
                   help="Comma-separated layer indices (default: all 13)")
    p.add_argument("--umap_layers",          default="0,4,8,12",
                   help="Subset of layers to generate UMAP plots for (expensive)")
    p.add_argument("--focus_phones",         default="TH,DH,V,F,S,Z,T,D",
                   help="Phones to highlight in focus-phone UMAP plots")
    p.add_argument("--max_utts_per_speaker", type=int, default=50)
    args = p.parse_args()

    print(f"=== Clustering  device={device} ===")

    registry      = get_model_registry(device)
    model_keys    = [k.strip()  for k in args.models.split(",")]
    layer_indices = [int(x)     for x in args.layers.split(",")]
    umap_layers   = [int(x)     for x in args.umap_layers.split(",")]
    focus_phones  = [p.strip().upper() for p in args.focus_phones.split(",")]

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key \'{key}\'. Available: {sorted(registry)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading utterances (split={args.split}) ...")
    utterances = load_probe_utterances(
        local_root           = args.data_root,
        split                = args.split,
        max_utts_per_speaker = args.max_utts_per_speaker,
    )
    print(f"  {len(utterances):,} utterances loaded")

    for key in model_keys:
        print(f"[Model: {key}]")
        model, processor = registry[key]["loader"]()
        cluster_model(
            model_key     = key,
            model         = model,
            processor     = processor,
            utterances    = utterances,
            layer_indices = layer_indices,
            umap_layers   = umap_layers,
            focus_phones  = focus_phones,
            device        = device,
            output_dir    = args.output_dir,
            split         = args.split,
        )
        del model
        torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    main()