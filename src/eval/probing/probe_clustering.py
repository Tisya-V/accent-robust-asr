"""
probe_clustering.py
Clustering metrics + UMAP visualisations for Whisper encoder representations.

Computes, per layer per model:
  - Silhouette score (phoneme labels, L1 labels)
  - Davies-Bouldin index (phoneme, L1)
  - Within-class / between-class distance ratio (phoneme, L1)
  - UMAP 2D projections saved as PNG

Usage:
    # Single model (parallelisable)
    python probe_clustering.py --models baseline  --split scripted
    python probe_clustering.py --models ctc_aux   --split scripted

    # Multiple models
    python probe_clustering.py --models baseline,baseline_lora,ctc_aux --split scripted

    # Custom layers / focus phones
    python probe_clustering.py --models ctc_aux --split scripted \
        --layers 0,3,6 --umap_layers 3,6 --focus_phones TH,DH,V,F,S,Z
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import umap as umap_lib

from src.eval.probing.probe_utils import (
    ARPABET_VOCAB, PHONE2ID,
    build_embedding_dataset, records_to_arrays, save_results,
)
from src.utils.load_l2arctic import load_probe_utterances
from src.utils.model_loader import get_model_registry
from src.config import LOCAL_L2ARCTIC_DIR, L1_GROUPS


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def clustering_metrics(X, labels, label_name) -> dict:
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) < len(unique) + 1:
        return {"error": "insufficient_data"}

    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_s, lab_s = X[idx], labels[idx]
    else:
        X_s, lab_s = X, labels

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_s)

    try:
        sil = float(silhouette_score(X_scaled, lab_s, metric="cosine",
                                     sample_size=min(2000, len(X_s))))
    except Exception:
        sil = float("nan")
    try:
        db  = float(davies_bouldin_score(X_scaled, lab_s))
    except Exception:
        db  = float("nan")
    try:
        ch  = float(calinski_harabasz_score(X_scaled, lab_s))
    except Exception:
        ch  = float("nan")

    try:
        centroids = {lbl: X_scaled[lab_s == lbl].mean(axis=0) for lbl in unique}
        within_dists = [
            np.mean(np.linalg.norm(X_scaled[lab_s == lbl] - centroids[lbl], axis=1))
            for lbl in unique if (lab_s == lbl).sum() >= 2
        ]
        within_mean = float(np.mean(within_dists)) if within_dists else float("nan")
        centroid_arr = np.stack(list(centroids.values()))
        n_c = len(centroid_arr)
        if n_c >= 2:
            diffs = centroid_arr[:, None, :] - centroid_arr[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            mask  = np.triu(np.ones((n_c, n_c), dtype=bool), k=1)
            between_mean = float(dists[mask].mean())
        else:
            between_mean = float("nan")
        wb_ratio = float(within_mean / between_mean) if between_mean > 0 else float("nan")
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

def make_umap_plot(X, labels, label_names, title, out_path,
                   max_points=3000, n_neighbors=15, min_dist=0.1, random_state=42):
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X, labels = X[idx], labels[idx]

    X_scaled  = StandardScaler().fit_transform(X)
    embedding = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                               n_components=2, metric="cosine",
                               random_state=random_state, verbose=False).fit_transform(X_scaled)

    n_classes = len(label_names)
    cmap      = colormaps["tab20" if n_classes <= 20 else "hsv"].resampled(n_classes)
    colors    = [cmap(i / n_classes) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() == 0: continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[colors[i]],
                   label=name, alpha=0.5, s=10, linewidths=0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2.5, fontsize=7, bbox_to_anchor=(1.01, 1),
              loc="upper left", borderaxespad=0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    UMAP saved → {out_path}")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def cluster_model(model_key, model, processor, utterances, layer_indices,
                      umap_layers, focus_phones, device, output_dir, split):
    """Run clustering + UMAP for a single model and save its own JSON."""
    out_json = Path(output_dir) / f"clustering_{model_key}_{split}.json"
    if out_json.exists():
        print(f"  [skip] {out_json} already exists — delete to re-run")
        return

    print(f"  Extracting hidden states …")
    records = build_embedding_dataset(
        model=model, processor=processor,
        utterances=utterances, layer_indices=layer_indices, device=device,
    )
    print(f"  {len(records)} records")

    layer_results = {}
    umap_dir      = Path(output_dir) / model_key

    for layer_idx in layer_indices:
        X, phone_ids, l1_ids, speakers = records_to_arrays(records, layer_idx)
        print(f"\n  Layer {layer_idx} | n={len(X)}")

        le       = LabelEncoder()
        spk_ints = le.fit_transform(speakers)

        lr = {
            "phoneme": clustering_metrics(X, phone_ids, "phoneme"),
            "l1":      clustering_metrics(X, l1_ids,    "L1/accent"),
            "speaker": clustering_metrics(X, spk_ints,  "speaker"),
        }
        layer_results[str(layer_idx)] = lr

        if layer_idx in umap_layers:
            base = f"umap_{model_key}_layer{layer_idx}"

            make_umap_plot(X, l1_ids, L1_GROUPS,
                title=f"{model_key} | Layer {layer_idx} | by L1",
                out_path=str(umap_dir / f"{base}_l1.png"))

            make_umap_plot(X, phone_ids, ARPABET_VOCAB,
                title=f"{model_key} | Layer {layer_idx} | by Phoneme",
                out_path=str(umap_dir / f"{base}_phoneme.png"))

            # Focus phones subset
            focus_ids = set(PHONE2ID.get(p, -1) for p in focus_phones) - {-1}
            mask_focus = np.isin(phone_ids, list(focus_ids))
            if mask_focus.sum() > 20:
                id_map        = {old: new for new, old in enumerate(sorted(focus_ids))}
                sorted_labels = [ARPABET_VOCAB[old] for old in sorted(focus_ids)]
                pid_remapped  = np.array([id_map[p] for p in phone_ids[mask_focus]])

                make_umap_plot(X[mask_focus], pid_remapped, sorted_labels,
                    title=f"{model_key} | Layer {layer_idx} | Focus phones",
                    out_path=str(umap_dir / f"{base}_focus_phones.png"))

                make_umap_plot(X[mask_focus], l1_ids[mask_focus], L1_GROUPS,
                    title=f"{model_key} | Layer {layer_idx} | Focus phones (by L1)",
                    out_path=str(umap_dir / f"{base}_focus_phones_l1.png"))

    save_results(layer_results, str(out_json))
    print(f"  Saved → {out_json}")

    # Summary
    print(f"  {'Layer':>6}  {'Phoneme sil':>14}  {'L1 sil':>10}")
    for li in layer_indices:
        ph_sil = layer_results[str(li)]["phoneme"].get("silhouette", float("nan"))
        l1_sil = layer_results[str(li)]["l1"].get("silhouette", float("nan"))
        print(f"  {li:>6}  {ph_sil:>14.3f}  {l1_sil:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",            default=LOCAL_L2ARCTIC_DIR)
    parser.add_argument("--models",               default="baseline",
                        help="Comma-separated model keys, e.g. baseline,ctc_aux")
    parser.add_argument("--split",                default="scripted",
                        choices=["scripted", "spontaneous", "all"])
    parser.add_argument("--output_dir",           default="results/clustering")
    parser.add_argument("--layers",               default="0,1,2,3,4,5,6")
    parser.add_argument("--umap_layers",          default="0,1,2,3,4,5,6",
                        help="Layers to generate UMAP plots for (expensive)")
    parser.add_argument("--focus_phones",         default="TH,DH,V,F,S,Z,T,D",
                        help="Comma-separated phones to highlight in a separate UMAP")
    parser.add_argument("--max_utts_per_speaker", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Clustering Probe === device={device}")

    layer_indices = [int(x) for x in args.layers.split(",")]
    umap_layers   = [int(x) for x in args.umap_layers.split(",")]
    focus_phones  = [p.strip().upper() for p in args.focus_phones.split(",")]
    model_keys    = [k.strip() for k in args.models.split(",")]
    registry      = get_model_registry(device)

    for key in model_keys:
        if key not in registry:
            raise ValueError(f"Unknown model key: '{key}'. Available: {list(registry.keys())}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading utterances (split={args.split}) …")
    utterances = load_probe_utterances(
        local_root=args.data_root, split=args.split,
        max_utts_per_speaker=args.max_utts_per_speaker, speakers={"all"},
    )
    print(f"    Found {len(utterances)} utterances")

    for key in model_keys:
        print(f"\n[Model: {key}]")
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

    print("\nAll done.")


if __name__ == "__main__":
    main()
