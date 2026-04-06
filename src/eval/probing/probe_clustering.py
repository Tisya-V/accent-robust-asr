"""
probe_clustering.py
Clustering metrics + UMAP visualisations for Whisper encoder representations.

Computes, per layer per model:
  - Silhouette score (phoneme labels, L1 labels)
  - Davies-Bouldin index (phoneme, L1)
  - Within-class / between-class distance ratio (phoneme, L1)
  - UMAP 2D projections saved as PNG:
      * Coloured by phoneme (all phones, or subset of interest)
      * Coloured by L1
      * Coloured by speaker

Usage:
    python probe_clustering.py \
        --data_root /path/to/l2arctic \
        --baseline_model openai/whisper-small \
        --lora_model   /path/to/lora-checkpoint \
        --split        scripted \
        --output_dir   results/clustering \
        [--max_utts 400] \
        [--layers 0,3,6] \
        [--focus_phones TH,DH,V,F,S,Z]
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from src.eval.probing.probe_utils import (
    ARPABET_VOCAB, PHONE2ID, L1_GROUPS,
    build_embedding_dataset, records_to_arrays, save_results,
)
from src.utils.load_l2arctic import load_probe_utterances

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from sklearn.preprocessing import LabelEncoder
import umap as umap_lib

# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def clustering_metrics(X: np.ndarray, labels: np.ndarray, label_name: str) -> dict:
    """
    Compute silhouette, Davies-Bouldin, Calinski-Harabasz, and
    within/between distance ratio for a set of embeddings and discrete labels.
    Subsamples to 5000 points max for speed.
    """
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) < len(unique) + 1:
        return {"error": "insufficient_data"}

    # Subsample for speed
    max_n = 5000
    if len(X) > max_n:
        idx = np.random.choice(len(X), max_n, replace=False)
        X_s, lab_s = X[idx], labels[idx]
    else:
        X_s, lab_s = X, labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_s)

    # Only run silhouette if labels have enough classes and samples per class
    try:
        sil = float(silhouette_score(X_scaled, lab_s, metric="cosine", sample_size=min(2000, len(X_s))))
    except Exception:
        sil = float("nan")

    try:
        db = float(davies_bouldin_score(X_scaled, lab_s))
    except Exception:
        db = float("nan")

    try:
        ch = float(calinski_harabasz_score(X_scaled, lab_s))
    except Exception:
        ch = float("nan")

    # Within/between distance ratio
    try:
        centroids = {}
        for lbl in unique:
            centroids[lbl] = X_scaled[lab_s == lbl].mean(axis=0)

        # Within-class: mean pairwise distance from centroid
        within_dists = []
        for lbl in unique:
            pts = X_scaled[lab_s == lbl]
            if len(pts) < 2: continue
            diffs = pts - centroids[lbl]
            within_dists.append(np.mean(np.linalg.norm(diffs, axis=1)))
        within_mean = float(np.mean(within_dists)) if within_dists else float("nan")

        # Between-class: mean pairwise centroid distance
        centroid_arr = np.stack(list(centroids.values()))
        n_c = len(centroid_arr)
        if n_c >= 2:
            diffs = centroid_arr[:, None, :] - centroid_arr[None, :, :]  # (n,n,D)
            dists = np.linalg.norm(diffs, axis=-1)
            mask = np.triu(np.ones((n_c, n_c), dtype=bool), k=1)
            between_mean = float(dists[mask].mean())
        else:
            between_mean = float("nan")

        wb_ratio = float(within_mean / between_mean) if between_mean > 0 else float("nan")

    except Exception:
        within_mean = between_mean = wb_ratio = float("nan")

    result = {
        "silhouette":    sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "within_mean":   within_mean,
        "between_mean":  between_mean,
        "wb_ratio":      wb_ratio,   # lower is better (compact clusters, wide separation)
        "n_samples":     int(len(X)),
        "n_classes":     int(len(unique)),
    }
    print(f"      {label_name:20s} | sil={sil:.3f}  DB={db:.3f}  W/B={wb_ratio:.3f}")
    return result


# ---------------------------------------------------------------------------
# UMAP plotting
# ---------------------------------------------------------------------------

def make_umap_plot(
    X: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    title: str,
    out_path: str,
    max_points: int = 3000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """
    Fit UMAP and save a 2D scatter plot coloured by `labels` (integer indices into label_names).
    """

    # Subsample for speed / readability
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X, labels = X[idx], labels[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
        verbose=False,
    )
    embedding = reducer.fit_transform(X_scaled)

    n_classes = len(label_names)
    cmap = cm.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() == 0: continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[i]], label=name,
            alpha=0.5, s=10, linewidths=0,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        markerscale=2.5, fontsize=7,
        bbox_to_anchor=(1.01, 1), loc="upper left",
        borderaxespad=0,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      UMAP saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",       default="data/l2_arctic")
    parser.add_argument("--baseline_model",  default="openai/whisper-small")
    parser.add_argument("--lora_model",      default="models/baseline_loraft")
    parser.add_argument("--split",           default="scripted",
                        choices=["scripted","spontaneous","all"])
    parser.add_argument("--output_dir",      default="results/clustering")
    parser.add_argument("--layers",          default="0,1,2,3,4,5,6")
    parser.add_argument("--max_utts",        type=int, default=None)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--umap_layers",     default="0,2,4,6",
                        help="Which layers to generate UMAP plots for (expensive)")
    parser.add_argument("--focus_phones",    default="TH,DH,V,F,S,Z,T,D",
                        help="Comma-separated phones to highlight in a separate UMAP")
    args = parser.parse_args()

    print("=== Probe Clustering Analysis ===")
    print(f"Running on device: {args.device}")

    layer_indices = [int(x) for x in args.layers.split(",")]
    umap_layers   = [int(x) for x in args.umap_layers.split(",")]
    focus_phones  = [p.strip().upper() for p in args.focus_phones.split(",")]

    print(f"\n[1/3] Loading probe utterances (split={args.split}) …")
    utterances = load_probe_utterances(
        local_root=args.data_root,
        split=args.split,
        max_utts=args.max_utts,
    )
    print(f"      {len(utterances)} utterances ready")

    models_to_eval = {"baseline": args.baseline_model}
    if args.lora_model:
        models_to_eval["lora"] = args.lora_model

    all_results = {}

    for model_name, model_path in models_to_eval.items():
        print(f"\n[2/3] Model: {model_name}")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        if model_name == "lora" and args.lora_model:
            base = PeftModel.from_pretrained(base, model_path)
            base = base.merge_and_unload()
        base = base.to(args.device)

        records = build_embedding_dataset(
            model=base, processor=processor,
            utterances=utterances, layer_indices=layer_indices,
            device=args.device,
        )
        del base

        print(f"[3/3] Computing clustering metrics + UMAP …")
        layer_results = {}
        out_dir = Path(args.output_dir) / model_name

        for layer_idx in layer_indices:
            X, phone_ids, l1_ids, speakers = records_to_arrays(records, layer_idx)
            print(f"\n  Layer {layer_idx} | n={len(X)}")

            lr = {}
            # Phoneme clustering
            lr["phoneme"] = clustering_metrics(X, phone_ids, "phoneme")
            # L1 clustering
            lr["l1"] = clustering_metrics(X, l1_ids, "L1/accent")

            # Unique speaker int IDs
            le = LabelEncoder()
            spk_ints = le.fit_transform(speakers)
            lr["speaker"] = clustering_metrics(X, spk_ints, "speaker")

            layer_results[str(layer_idx)] = lr

            # UMAP plots for selected layers
            if layer_idx in umap_layers:
                # Coloured by L1
                make_umap_plot(
                    X, l1_ids, L1_GROUPS,
                    title=f"{model_name} | Layer {layer_idx} | Coloured by L1",
                    out_path=str(out_dir / f"umap_layer{layer_idx}_l1.png"),
                )
                # Coloured by phoneme (all)
                make_umap_plot(
                    X, phone_ids, ARPABET_VOCAB,
                    title=f"{model_name} | Layer {layer_idx} | Coloured by Phoneme",
                    out_path=str(out_dir / f"umap_layer{layer_idx}_phoneme.png"),
                )
                # Focus phones subset
                focus_ids = set(PHONE2ID.get(p, -1) for p in focus_phones) - {-1}
                mask_focus = np.isin(phone_ids, list(focus_ids))
                if mask_focus.sum() > 20:
                    focus_labels_subset = [p for p in focus_phones if PHONE2ID.get(p, -1) in focus_ids]
                    X_f = X[mask_focus]
                    pid_f = phone_ids[mask_focus]
                    # Remap ids to 0..len(focus)
                    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(focus_ids))}
                    pid_f_remapped = np.array([id_map[p] for p in pid_f])
                    sorted_labels = [ARPABET_VOCAB[old] for old in sorted(focus_ids)]
                    make_umap_plot(
                        X_f, pid_f_remapped, sorted_labels,
                        title=f"{model_name} | Layer {layer_idx} | Focus phones",
                        out_path=str(out_dir / f"umap_layer{layer_idx}_focus_phones.png"),
                    )
                    # Also colour by L1 within focus phones
                    make_umap_plot(
                        X_f, l1_ids[mask_focus], L1_GROUPS,
                        title=f"{model_name} | Layer {layer_idx} | Focus phones (by L1)",
                        out_path=str(out_dir / f"umap_layer{layer_idx}_focus_phones_l1.png"),
                    )

        all_results[model_name] = layer_results

    # Save JSON
    out_json = Path(args.output_dir) / f"clustering_{args.split}.json"
    save_results(all_results, str(out_json))
    print(f"\nDone. Results → {out_json}")

    # Print summary
    print("\n=== Summary: Phoneme Silhouette by Layer ===")
    header = f"{'Layer':>6}" + "".join(f"  {m:>14}" for m in all_results)
    print(header)
    for li in layer_indices:
        row = f"{li:>6}"
        for m in all_results:
            val = all_results[m].get(str(li), {}).get("phoneme", {}).get("silhouette", float("nan"))
            row += f"  {val:>14.3f}"
        print(row)

    print("\n=== Summary: L1 Silhouette by Layer ===")
    print(header)
    for li in layer_indices:
        row = f"{li:>6}"
        for m in all_results:
            val = all_results[m].get(str(li), {}).get("l1", {}).get("silhouette", float("nan"))
            row += f"  {val:>14.3f}"
        print(row)


if __name__ == "__main__":
    main()

