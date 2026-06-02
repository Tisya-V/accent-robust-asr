import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/whisper_baseline/whisper_baseline.csv")

stats = (df.dropna(subset=["wer"])
           .groupby("l1")["wer"]
           .agg(mean="mean", sem=lambda x: x.sem(), median="median",
                std="std", count="count"))

order = sorted(l for l in stats.index if l != "English") + ["English"]
stats = stats.loc[order]

# --- bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(stats.index, stats["mean"], width=0.6)

ax.set_ylabel("Word Error Rate (WER)")
ax.set_xlabel("Speaker L1")
ax.set_ylim(0, stats["mean"].max() * 1.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Whisper baseline WER by speaker L1")

for bar, (_, row) in zip(bars, stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{row['mean']:.1%}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("results/whisper_baseline/wer_by_l1.pdf", bbox_inches="tight")
plt.savefig("results/whisper_baseline/wer_by_l1.png", dpi=150, bbox_inches="tight")

# --- summary table ---
out = stats.copy()
out.index.name = "L1"
out.columns = ["Mean WER", "SEM", "Median WER", "Std", "N"]
out.to_csv("results/whisper_baseline/wer_summary.csv", float_format="%.4f")
out.to_latex("results/whisper_baseline/wer_summary.tex",
             float_format="%.4f", caption="Whisper baseline WER by speaker L1.",
             label="tab:whisper_baseline_wer")

print("saved")
