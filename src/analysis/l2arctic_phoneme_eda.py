# %% [markdown]
# # Phoneme Replacement Structure in L2-ARCTIC
#
# Goal:
# Show that phoneme substitutions are not random, but are related to
# articulatory-feature similarity.
#
# Main outputs:
# 1. Substitution confusion heatmap
# 2. Observed substitution-distance distribution
# 3. Observed vs null distance comparison
# 4. Distance-bin substitution plot
# 5. Feature-retention / feature-change summaries
# 6. Canonical-phone "family" summaries for realised substitutions
#
# Notes:
# - Correct phones are kept as type='c'
# - add/del are collapsed into substitution-style events using SIL
# - SIL is excluded from articulatory-distance calculations
# - PanPhon feature vectors are used for feature-wise analysis
#
# This script is designed to be thesis-friendly:
# it prioritizes interpretable plots over overly clever pair-level regressions.

# %%
import os
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from textgrid import TextGrid, IntervalTier
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr, pearsonr

import panphon
from phonecodes import phonecodes

from src.utils.load_l2arctic import load_all_scripted

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

# %%
OUT_DIR = Path("./phoneme_replacement_structure_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIL = "SIL"

# %% [markdown]
# ## Load data

# %%
dataset = load_all_scripted()
dataset_df = pd.DataFrame(dataset)

dataset_df = dataset_df[
    dataset_df["annotation"].notnull() &
    (dataset_df["annotation"] != "None")
].copy()

print(f"Loaded {len(dataset_df)} annotated utterances.")
print(dataset_df[["utterance_id", "speaker", "l1", "text", "annotation"]].head())

# %% [markdown]
# ## Helpers

# %%
def clean_phone(p):
    """
    Normalize raw phone token:
    - uppercase
    - remove whitespace, trailing punctuation, stress/index markers, star
    - collapse pause/silence variants to SIL
    """
    if pd.isna(p):
        return None

    s = str(p).strip().upper()
    s = s.rstrip(",")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[\d\*]+$", "", s)

    if s in {"", "ERR", "NONE"}:
        return None
    if s in {"SP", "SIL", "PAU", "SPN", "NSN"}:
        return SIL

    return s


def parse_phone_tier(tier: IntervalTier):
    """
    Parse phone tier entries of form: 'CPL,PPL,type'

    New conventions:
    - c: correct, force ref == hyp
    - s: substitution, keep as-is
    - d: deletion -> substitution to SIL
    - a: addition -> substitution from SIL
    """
    rows = []

    for interval in tier:
        mark = interval.mark.strip()
        if not mark:
            continue

        parts = [x.strip() for x in mark.split(",")]
        if len(parts) != 3:
            continue

        cpl, ppl, typ = parts
        ref = clean_phone(cpl)
        hyp = clean_phone(ppl)
        typ = typ.lower()

        if typ == "c":
            base = ref if ref is not None else hyp
            if base is None:
                continue
            ref = base
            hyp = base
            out_type = "c"

        elif typ == "s":
            if ref is None or hyp is None:
                continue
            out_type = "s"

        elif typ == "d":
            if ref is None:
                continue
            hyp = SIL
            out_type = "s"

        elif typ == "a":
            if hyp is None:
                continue
            ref = SIL
            out_type = "s"

        else:
            continue

        rows.append({
            "start": interval.minTime,
            "end": interval.maxTime,
            "ref": ref,
            "hyp": hyp,
            "type": out_type,
            "orig_type": typ
        })

    return rows


def safe_load_textgrid(file_path):
    try:
        tg = TextGrid.fromFile(file_path)
        global_xmax = tg.maxTime
        for tier in tg:
            if hasattr(tier, "maxTime") and tier.maxTime > global_xmax:
                tier.maxTime = global_xmax
            if hasattr(tier, "xmin") and tier.xmin < tg.xmin:
                tier.xmin = tg.xmin
        return tg
    except Exception as e:
        print(f"Failed {file_path}: {e}")
        return None


def load_phone_events_from_textgrids(df, annt_col="annotation"):
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        annt_file = row[annt_col]
        if pd.isna(annt_file) or not os.path.exists(annt_file):
            continue

        tg = safe_load_textgrid(annt_file)
        if tg is None:
            continue

        phone_tier = None
        for tier in tg:
            if tier.name.strip().lower() in {"phones", "phone"}:
                phone_tier = tier
                break
        if phone_tier is None and len(tg) > 1:
            phone_tier = tg[1]

        if phone_tier is None:
            continue

        phone_rows = parse_phone_tier(phone_tier)

        for r in phone_rows:
            r["speaker"] = row.get("speaker", "unknown")
            r["l1"] = row.get("l1", "unknown")
            r["utt_id"] = row.get("utterance_id", idx)
            r["text"] = row.get("text", "")

        rows.extend(phone_rows)

    return pd.DataFrame(rows)


def clean_arpabet(p):
    p = clean_phone(p)
    if p is None or p == SIL:
        return None
    return p


def arpabet_to_ipa_single(p):
    p = clean_arpabet(p)
    if p is None:
        return None

    try:
        ipa = phonecodes.arpabet2ipa(p, "eng")
    except Exception:
        try:
            ipa = phonecodes.convert(p, "arpabet", "ipa", "eng")
        except Exception:
            return None

    if ipa is None:
        return None

    ipa = ipa.strip().replace("ˈ", "").replace("ˌ", "").replace(" ", "")
    return ipa or None


# %%
df_phones = load_phone_events_from_textgrids(dataset_df)

print(df_phones.head())
print(df_phones["type"].value_counts(dropna=False))
print(df_phones["orig_type"].value_counts(dropna=False))

# %% [markdown]
# ## Prepare substitution table

# %%
all_events = df_phones.copy()

subs = all_events[
    (all_events["type"] == "s") &
    (all_events["ref"] != all_events["hyp"])
].copy()

subs["ref_clean"] = subs["ref"].map(clean_arpabet)
subs["hyp_clean"] = subs["hyp"].map(clean_arpabet)

# Keep segmental substitutions only for articulatory analysis
subs_seg = subs.dropna(subset=["ref_clean", "hyp_clean"]).copy()

print(f"All substitution-like events (incl SIL-based add/del): {len(subs)}")
print(f"Segmental substitutions with clean phones: {len(subs_seg)}")

# %% [markdown]
# ## Build PanPhon feature table once

# %%
ft = panphon.FeatureTable()

phones_for_features = sorted(set(subs_seg["ref_clean"]) | set(subs_seg["hyp_clean"]))

feature_rows = []
for p in phones_for_features:
    ipa = arpabet_to_ipa_single(p)
    if ipa is None:
        continue
    try:
        segs = ft.word_to_vector_list(ipa, numeric=True)
        names = ft.names
        if segs:
            vec = np.array(segs[0], dtype=float)
            feature_rows.append({
                "phone": p,
                "ipa": ipa,
                "vec": vec
            })
    except Exception:
        pass

feat_map = {r["phone"]: r["vec"] for r in feature_rows}
ipa_map = {r["phone"]: r["ipa"] for r in feature_rows}
feature_names = list(ft.names)

print(f"Phones with feature vectors: {len(feat_map)}")
print("Example features:", feature_names[:10])

# %%
subs_seg = subs_seg[
    subs_seg["ref_clean"].isin(feat_map) &
    subs_seg["hyp_clean"].isin(feat_map)
].copy()

# Pair counts
pair_counts = (
    subs_seg.groupby(["ref_clean", "hyp_clean"])
    .size()
    .reset_index(name="count")
)

# Canonical counts over all outcomes (correct + substitution), segmental only
all_events["ref_clean"] = all_events["ref"].map(clean_arpabet)
all_events["hyp_clean"] = all_events["hyp"].map(clean_arpabet)

canonical_totals = (
    all_events[all_events["ref_clean"].isin(feat_map)]
    .groupby("ref_clean")
    .size()
)

# Distance and feature summary per pair
pair_rows = []
for _, row in pair_counts.iterrows():
    ref = row["ref_clean"]
    hyp = row["hyp_clean"]
    ref_vec = feat_map[ref]
    hyp_vec = feat_map[hyp]

    absdiff = np.abs(ref_vec - hyp_vec)
    changed = absdiff > 0

    pair_rows.append({
        "ref": ref,
        "hyp": hyp,
        "ref_ipa": ipa_map[ref],
        "hyp_ipa": ipa_map[hyp],
        "count": int(row["count"]),
        "dist_L1": float(absdiff.sum()),
        "n_changed_features": int(changed.sum()),
        "pair_prob_all_ref": float(row["count"] / canonical_totals[ref]) if canonical_totals[ref] > 0 else np.nan
    })

pair_df = pd.DataFrame(pair_rows)
pair_df = pair_df.sort_values("count", ascending=False)

print(f"{len(pair_df)} unique segmental substitution pairs")
display(pair_df.head(20))

pair_df.to_csv(OUT_DIR / "substitution_pairs_with_distance.csv", index=False)

# %% [markdown]
# ## Plot 1: substitution confusion heatmap
#
# This is the best first figure:
# it shows that substitutions are structured, not diffuse/random.

# %%
phones = sorted(set(pair_df["ref"]) | set(pair_df["hyp"]))

cm = confusion_matrix(
    pair_df.loc[pair_df.index.repeat(pair_df["count"]), "ref"],
    pair_df.loc[pair_df.index.repeat(pair_df["count"]), "hyp"],
    labels=phones
)

cm_df = pd.DataFrame(cm, index=phones, columns=phones)

conf_mass = cm_df.sum(axis=1) + cm_df.sum(axis=0)
phone_order = conf_mass.sort_values(ascending=False).index.tolist()
cm_ord = cm_df.loc[phone_order, phone_order]

K = min(25, len(cm_ord))
cm_top = cm_ord.iloc[:K, :K].copy()
for i in range(min(cm_top.shape)):
    cm_top.iat[i, i] = np.nan
cm_top = cm_top.replace(0, np.nan)

plt.figure(figsize=(14, 11))
ax = sns.heatmap(
    cm_top,
    cmap="Greens",
    square=True,
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Substitution count"}
)
ax.set_title(f"Top {K} Substitution Confusions", pad=12)
ax.set_xlabel("Realised phone")
ax.set_ylabel("Canonical phone")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_substitution_confusion_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Plot 2: observed distance distribution
#
# If substitutions are constrained by articulatory similarity, observed
# substitutions should be concentrated at lower feature distances.

# %%
token_level_dist = pair_df.loc[pair_df.index.repeat(pair_df["count"]), "dist_L1"].reset_index(drop=True)
unique_pair_dist = pair_df["dist_L1"].copy()

plt.figure(figsize=(10, 6))
sns.histplot(token_level_dist, bins=20, kde=True, color="#4C72B0", alpha=0.75)
plt.xlabel("Articulatory feature distance (L1 over PanPhon vectors)")
plt.ylabel("Number of substitution tokens")
plt.title("Observed Distribution of Substitution Distances")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_observed_distance_distribution_tokens.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(unique_pair_dist, bins=20, kde=True, color="#55A868", alpha=0.75)
plt.xlabel("Articulatory feature distance (L1 over PanPhon vectors)")
plt.ylabel("Number of unique substitution pairs")
plt.title("Observed Distribution of Unique Substitution-Pair Distances")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_observed_distance_distribution_unique_pairs.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Plot 3: observed vs null distance distribution
#
# Strongest figure for the non-randomness claim.
#
# Null:
# - keep the same phone inventory
# - sample random ref-hyp phone pairs (excluding identity)
# - compare their distances to the observed substitution-pair distances
#
# This asks:
# Are observed substitution pairs closer than arbitrary cross-phone pairs?

# %%
inventory = sorted(set(pair_df["ref"]) | set(pair_df["hyp"]))

all_possible_pairs = []
for ref in inventory:
    for hyp in inventory:
        if ref != hyp:
            if ref in feat_map and hyp in feat_map:
                dist = np.abs(feat_map[ref] - feat_map[hyp]).sum()
                all_possible_pairs.append((ref, hyp, dist))

all_possible_df = pd.DataFrame(all_possible_pairs, columns=["ref", "hyp", "dist_L1"])

observed_unique = pair_df["dist_L1"].to_numpy()
rng = np.random.default_rng(42)

null_unique = all_possible_df.sample(
    n=len(observed_unique),
    replace=False if len(all_possible_df) >= len(observed_unique) else True,
    random_state=42
)["dist_L1"].to_numpy()

ks_stat, ks_p = ks_2samp(observed_unique, null_unique)
mw_stat, mw_p = mannwhitneyu(observed_unique, null_unique, alternative="less")

plt.figure(figsize=(10, 6))
sns.kdeplot(observed_unique, fill=True, label="Observed substitution pairs", color="#4C72B0", alpha=0.5)
sns.kdeplot(null_unique, fill=True, label="Random phone pairs (null)", color="#C44E52", alpha=0.35)
plt.xlabel("Articulatory feature distance (L1)")
plt.ylabel("Density")
plt.title(
    "Observed vs Null Distance Distribution\n"
    f"KS p={ks_p:.3g}, Mann-Whitney(one-sided) p={mw_p:.3g}"
)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "04_observed_vs_null_distance_kde.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional boxplot version
plt.figure(figsize=(8, 6))
box_df = pd.DataFrame({
    "distance": np.concatenate([observed_unique, null_unique]),
    "group": (["Observed"] * len(observed_unique)) + (["Null"] * len(null_unique))
})
sns.boxplot(data=box_df, x="group", y="distance", palette=["#4C72B0", "#C44E52"])
sns.stripplot(data=box_df.sample(min(800, len(box_df)), random_state=42),
              x="group", y="distance", color="black", alpha=0.15, size=3)
plt.title("Observed substitution pairs are closer than random pairs")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_observed_vs_null_distance_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Plot 4: distance-bin substitution trend
#
# This is a cleaner version of the "distance relationship" idea:
# - x = distance bin
# - y = number of observed unique substitution pairs in that bin
#
# It avoids misleading pair-probability interpretations.

# %%
bin_width = 1.0
bins = np.arange(0, max(pair_df["dist_L1"].max(), all_possible_df["dist_L1"].max()) + bin_width, bin_width)

pair_df["dist_bin"] = pd.cut(pair_df["dist_L1"], bins=bins, include_lowest=True)
all_possible_df["dist_bin"] = pd.cut(all_possible_df["dist_L1"], bins=bins, include_lowest=True)

obs_bin = pair_df.groupby("dist_bin").size().reset_index(name="observed_unique_pairs")
null_bin = all_possible_df.groupby("dist_bin").size().reset_index(name="all_possible_pairs")

dist_bin_df = obs_bin.merge(null_bin, on="dist_bin", how="left")
dist_bin_df["dist_mid"] = dist_bin_df["dist_bin"].apply(lambda x: x.mid)
dist_bin_df["obs_over_possible"] = dist_bin_df["observed_unique_pairs"] / dist_bin_df["all_possible_pairs"]

# Correlations
valid = dist_bin_df.dropna(subset=["dist_mid", "obs_over_possible"])
pear_r, pear_p = pearsonr(valid["dist_mid"], valid["obs_over_possible"])
spear_rho, spear_p = spearmanr(valid["dist_mid"], valid["obs_over_possible"])

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=dist_bin_df,
    x="dist_mid",
    y="obs_over_possible",
    s=80,
    color="navy"
)
sns.regplot(
    data=dist_bin_df,
    x="dist_mid",
    y="obs_over_possible",
    scatter=False,
    color="red",
    line_kws={"lw": 2}
)
plt.xlabel("Articulatory feature distance bin midpoint")
plt.ylabel("Observed substitution pairs / all possible pairs")
plt.title(
    "Substitution likelihood by articulatory distance bin\n"
    f"Pearson r={pear_r:.3f}, Spearman rho={spear_rho:.3f}"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_distance_bin_ratio_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

dist_bin_df.to_csv(OUT_DIR / "distance_bin_counts_and_ratios.csv", index=False)

# %% [markdown]
# ## Plot 5: feature retention and change
#
# This directly addresses whether substitutions preserve many articulatory
# properties. For each feature dimension:
# - retention rate = proportion of substitution tokens where ref and hyp
#   have the same value on that feature
# - change rate = proportion where that feature differs

# %%
feature_change_rows = []

for _, row in pair_df.iterrows():
    ref = row["ref"]
    hyp = row["hyp"]
    count = row["count"]
    ref_vec = feat_map[ref]
    hyp_vec = feat_map[hyp]

    for i, fname in enumerate(feature_names):
        same = int(ref_vec[i] == hyp_vec[i])
        feature_change_rows.append({
            "feature": fname,
            "same": same,
            "different": 1 - same,
            "weight": count
        })

feature_change_df = pd.DataFrame(feature_change_rows)

feature_summary = (
    feature_change_df.groupby("feature")
    .apply(lambda g: pd.Series({
        "same_weighted": np.average(g["same"], weights=g["weight"]),
        "different_weighted": np.average(g["different"], weights=g["weight"])
    }))
    .reset_index()
    .sort_values("same_weighted", ascending=False)
)

display(feature_summary.head(20))
feature_summary.to_csv(OUT_DIR / "feature_retention_summary.csv", index=False)

plt.figure(figsize=(10, 8))
plot_df = feature_summary.sort_values("same_weighted", ascending=True)
plt.barh(plot_df["feature"], plot_df["same_weighted"], color="#4C72B0")
plt.xlabel("Weighted retention rate across substitution tokens")
plt.ylabel("PanPhon feature")
plt.title("Feature Retention in Observed Substitutions")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_feature_retention_barh.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 8))
plot_df = feature_summary.sort_values("different_weighted", ascending=True)
plt.barh(plot_df["feature"], plot_df["different_weighted"], color="#C44E52")
plt.xlabel("Weighted change rate across substitution tokens")
plt.ylabel("PanPhon feature")
plt.title("Feature Change in Observed Substitutions")
plt.tight_layout()
plt.savefig(OUT_DIR / "08_feature_change_barh.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Plot 6: canonical-phone family summaries
#
# For each canonical phone:
# - how many distinct realised substitutions does it have?
# - how dispersed are those realised phones in feature space?
#
# This supports the claim that substitution families are structured.

# %%
family_rows = []

for ref_phone, group in subs_seg.groupby("ref_clean"):
    realised_counts = group["hyp_clean"].value_counts()
    realised_probs = realised_counts / realised_counts.sum()

    realised_phones = realised_counts.index.tolist()
    realised_weights = realised_counts.values.astype(float)

    # weighted mean distance from canonical phone
    dists_from_ref = [np.abs(feat_map[ref_phone] - feat_map[h]).sum() for h in realised_phones]
    mean_dist_from_ref = np.average(dists_from_ref, weights=realised_weights)

    # weighted pairwise dispersion among realised phones
    pair_dists = []
    pair_w = []
    for i in range(len(realised_phones)):
        for j in range(i + 1, len(realised_phones)):
            d = np.abs(feat_map[realised_phones[i]] - feat_map[realised_phones[j]]).sum()
            w = realised_weights[i] * realised_weights[j]
            pair_dists.append(d)
            pair_w.append(w)

    mean_dispersion = np.average(pair_dists, weights=pair_w) if len(pair_dists) > 0 else np.nan

    family_rows.append({
        "ref": ref_phone,
        "total_sub_tokens": int(realised_counts.sum()),
        "n_distinct_realised_subs": int(len(realised_phones)),
        "dominant_realisation": realised_counts.index[0],
        "dominant_share": float(realised_probs.iloc[0]),
        "mean_distance_from_ref": float(mean_dist_from_ref),
        "mean_pairwise_dispersion": float(mean_dispersion) if not np.isnan(mean_dispersion) else np.nan
    })

family_df = pd.DataFrame(family_rows).sort_values("total_sub_tokens", ascending=False)
display(family_df.head(20))
family_df.to_csv(OUT_DIR / "canonical_phone_substitution_families.csv", index=False)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=family_df,
    x="mean_distance_from_ref",
    y="n_distinct_realised_subs",
    size="total_sub_tokens",
    hue="dominant_share",
    palette="viridis",
    sizes=(40, 400),
    alpha=0.8
)
for _, row in family_df.nlargest(12, "total_sub_tokens").iterrows():
    plt.text(
        row["mean_distance_from_ref"] + 0.03,
        row["n_distinct_realised_subs"] + 0.03,
        row["ref"],
        fontsize=9
    )
plt.xlabel("Mean distance from canonical phone")
plt.ylabel("Number of distinct realised substitutions")
plt.title("Canonical-phone substitution families")
plt.tight_layout()
plt.savefig(OUT_DIR / "09_canonical_phone_family_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary tables

# %%
summary_stats = pd.DataFrame([{
    "n_annotated_utterances": len(dataset_df),
    "n_all_phone_events": len(df_phones),
    "n_substitution_events_incl_sil": len(subs),
    "n_segmental_substitution_events": len(subs_seg),
    "n_unique_segmental_sub_pairs": len(pair_df),
    "mean_observed_unique_pair_distance": observed_unique.mean(),
    "mean_null_unique_pair_distance": null_unique.mean(),
    "ks_stat": ks_stat,
    "ks_p": ks_p,
    "mannwhitney_less_stat": mw_stat,
    "mannwhitney_less_p": mw_p,
    "distance_bin_pearson_r": pear_r,
    "distance_bin_pearson_p": pear_p,
    "distance_bin_spearman_rho": spear_rho,
    "distance_bin_spearman_p": spear_p
}])

summary_stats.to_csv(OUT_DIR / "summary_stats.csv", index=False)
df_phones.to_csv(OUT_DIR / "phone_events_full.csv", index=False)
subs_seg.to_csv(OUT_DIR / "segmental_substitution_tokens.csv", index=False)

print(f"Saved outputs to: {OUT_DIR.resolve()}")