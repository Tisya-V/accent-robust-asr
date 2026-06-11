"""Aggregate corpus-level WER/MER deltas vs Whisper baseline across training seeds.

Reproduces the corpus_wer_mer.csv computation from notebooks/compare_evals.ipynb
(cell 4) per-seed, then reports mean +/- stdev across seeds for each config.
"""
import jiwer
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/vol/gpudata/tsv22-fyp/accent-robust-asr")
BRIDGE_DIR = ROOT / "results" / "bridge_eval"
BASELINE_CSV = ROOT / "results" / "whisper_baseline" / "whisper_baseline.csv"
SAVE_DIR = ROOT / "results" / "bridge_eval" / "summary_tables" / "all"

CONFIGS = {
    "bridge_dtw_fixed_eps_0.3_ode_renorm": {
        42: "bridge_dtw_fixed_eps_0.3_ode_renorm.csv",
        4: "bridge_dtw_fixed_eps_0.3_seed4_ode_renorm.csv",
        67: "bridge_dtw_fixed_eps_0.3_seed67_ode_renorm.csv",
    },
    "bridge_dtw_fixed_eps_0.5_ode_renorm": {
        42: "bridge_dtw_fixed_eps_0.5_ode_renorm.csv",
        4: "bridge_dtw_fixed_eps_0.5_seed4_ode_renorm.csv",
        67: "bridge_dtw_fixed_eps_0.5_seed67_ode_renorm.csv",
    },
    # x0 variants — kept here even though they don't beat baseline, to
    # document that they were evaluated across seeds rather than abandoned
    # after a single (high-variance) run.
    "bridge_dtw_fixed_x0_0.3_ode_renorm": {
        42: "bridge_dtw_fixed_x0_0.3_ode_renorm.csv",
        4: "bridge_dtw_fixed_x0_0.3_seed4_ode_renorm.csv",
        67: "bridge_dtw_fixed_x0_0.3_seed67_ode_renorm.csv",
    },
    "bridge_dtw_fixed_x0_0.5_ode_renorm": {
        42: "bridge_dtw_fixed_x0_0.5_ode_renorm.csv",
        4: "bridge_dtw_fixed_x0_0.5_seed4_ode_renorm.csv",
        67: "bridge_dtw_fixed_x0_0.5_seed67_ode_renorm.csv",
    },
}


def corpus_wer_mer_delta(bridge_csv: Path, baseline_df: pd.DataFrame) -> dict:
    bridge_df = pd.read_csv(bridge_csv)
    m = bridge_df[["utterance_id", "speaker", "reference_norm", "prediction_norm"]].merge(
        baseline_df[["utterance_id", "speaker", "prediction_norm"]].rename(
            columns={"prediction_norm": "pred_base"}),
        on=["utterance_id", "speaker"], how="inner")

    refs = m["reference_norm"].fillna("").tolist()
    preds_base = m["pred_base"].fillna("").tolist()
    preds_bridge = m["prediction_norm"].fillna("").tolist()

    wer_base = jiwer.wer(refs, preds_base) * 100
    wer_bridge = jiwer.wer(refs, preds_bridge) * 100
    mer_base = jiwer.mer(refs, preds_base) * 100
    mer_bridge = jiwer.mer(refs, preds_bridge) * 100

    return {
        "n": len(m),
        "wer_baseline_%": wer_base, "wer_bridge_%": wer_bridge, "wer_delta_%": wer_bridge - wer_base,
        "mer_baseline_%": mer_base, "mer_bridge_%": mer_bridge, "mer_delta_%": mer_bridge - mer_base,
    }


def main():
    baseline_df = pd.read_csv(BASELINE_CSV)

    per_seed_rows = []
    summary_rows = []
    for config_name, seed_files in CONFIGS.items():
        print(f"\n=== {config_name} ===")
        wer_deltas, mer_deltas = [], []
        for seed, fname in seed_files.items():
            csv_path = BRIDGE_DIR / fname
            if not csv_path.exists():
                print(f"  seed {seed:>3}: skipping, missing {csv_path}")
                continue
            stats = corpus_wer_mer_delta(csv_path, baseline_df)
            wer_deltas.append(stats["wer_delta_%"])
            mer_deltas.append(stats["mer_delta_%"])
            per_seed_rows.append({"model": config_name, "seed": seed, **stats})
            print(f"  seed {seed:>3}: n={stats['n']}  "
                  f"wer_baseline_%={stats['wer_baseline_%']:.3f}  wer_bridge_%={stats['wer_bridge_%']:.3f}  "
                  f"wer_delta_%={stats['wer_delta_%']:+.3f}  mer_delta_%={stats['mer_delta_%']:+.3f}")

        if len(wer_deltas) < 2:
            print(f"  fewer than 2 seeds available ({len(wer_deltas)}) — skipping mean/std")
            continue

        wer_arr, mer_arr = np.array(wer_deltas), np.array(mer_deltas)
        summary_rows.append({
            "model": config_name,
            "n_seeds": len(wer_deltas),
            "wer_delta_mean_%": wer_arr.mean(), "wer_delta_std_%": wer_arr.std(ddof=1),
            "mer_delta_mean_%": mer_arr.mean(), "mer_delta_std_%": mer_arr.std(ddof=1),
        })
        print(f"  {'mean':>8}: wer_delta_%={wer_arr.mean():+.3f} +/- {wer_arr.std(ddof=1):.3f}   "
              f"mer_delta_%={mer_arr.mean():+.3f} +/- {mer_arr.std(ddof=1):.3f}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_df = pd.DataFrame(per_seed_rows).round(3)
    summary_df = pd.DataFrame(summary_rows).round(3)
    per_seed_df.to_csv(SAVE_DIR / "multiseed_wer_per_seed.csv", index=False)
    summary_df.to_csv(SAVE_DIR / "multiseed_wer_summary.csv", index=False)
    print(f"\nSaved {SAVE_DIR / 'multiseed_wer_per_seed.csv'}")
    print(f"Saved {SAVE_DIR / 'multiseed_wer_summary.csv'}")


if __name__ == "__main__":
    main()
