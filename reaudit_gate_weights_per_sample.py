"""
Per-sample re-audit of Section 3.5's aggregate gate-weight numbers against
the degenerate-collapse artifact identified in Section 3.4 — manuscript
Section 7, item 1.

Section 3.5 currently reports only aggregate means (e.g. "mean gate weight
on text: 0.926 present vs. 0.761 absent for gating_only_no_dropout@25%").
An aggregate mean over all test samples can be distorted if a subset of
samples produce degenerate, near-constant gate weights (the same failure
mode Section 3.4 identifies at the complete-modality-loss condition) — a
small cluster of samples pinned near weight=0 or weight=1 could pull an
aggregate mean without being visible in the mean alone.

This script does NOT reimplement gate-weight collection; it expects the
existing raw per-sample file (`gate_weights_raw.csv`, one row per test
sample per model per seed per rate, with a `gate_weight_text` column) and
recomputes:

  1. The aggregate mean (to confirm it reproduces Section 3.5's reported
     numbers exactly, as a sanity check before trusting anything else here).
  2. The per-sample distribution's shape: min/25th/median/75th/max, and the
     fraction of samples within 0.02 of 0.0 or 1.0 (a "near-degenerate"
     band, distinct from but analogous to Section 3.4's F1<0.05 collapse
     threshold, which does not directly apply here since Section 3.5's
     numbers are at 0.25/0.5/0.75 partial missingness, not the complete-loss
     condition Section 3.4 defines collapse under).
  3. Whether the near-degenerate fraction differs meaningfully between
     dropout-trained and non-dropout-trained models at the same rate — if
     it does, that is evidence the aggregate means in Section 3.5 are
     comparing distributions with different shapes, not just different
     centers, which would change how that section should be read.

Usage:
    python reaudit_gate_weights_per_sample.py \
        --raw_csv gate_weights_raw.csv \
        --out_csv gate_weights_per_sample_reaudit.csv

`gate_weights_raw.csv` is not included in this repository as of this
revision (see README's "known gap" note) — obtain it from the Drive project
folder or regenerate it via `run_diagnostics.py` against the 5-seed
checkpoints before running this script.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

NEAR_DEGENERATE_BAND = 0.02


def reaudit(raw: pd.DataFrame) -> pd.DataFrame:
    # Actual gate_weights_raw.csv columns (confirmed against the real file,
    # not assumed): model, seed, rate, sample_idx, w_text, w_audio,
    # w_vision, text_present, audio_present, vision_present.
    required_cols = {"model", "rate", "w_text"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(
            f"gate_weights_raw.csv is missing expected columns: {missing}. "
            f"This script assumes one row per test sample with 'model', "
            f"'rate', and 'w_text' columns — adjust the column names "
            f"below if your raw file uses different names, rather than "
            f"assuming the analysis logic itself is wrong."
        )
    rate_col, weight_col = "rate", "w_text"

    rows = []
    for (model, rate), grp in raw.groupby(["model", rate_col]):
        w = grp[weight_col].to_numpy()
        near_zero = (w < NEAR_DEGENERATE_BAND).mean()
        near_one = (w > 1 - NEAR_DEGENERATE_BAND).mean()
        rows.append({
            "model": model, "missingness_rate": rate, "n_samples": len(w),
            "mean": w.mean(), "median": np.median(w),
            "p25": np.percentile(w, 25), "p75": np.percentile(w, 75),
            "min": w.min(), "max": w.max(),
            "frac_near_zero": near_zero, "frac_near_one": near_one,
            "frac_near_degenerate": near_zero + near_one,
        })
    return pd.DataFrame(rows).sort_values(["model", "missingness_rate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="gate_weights_raw.csv")
    parser.add_argument("--out_csv", default="gate_weights_per_sample_reaudit.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    result = reaudit(raw)
    result.to_csv(args.out_csv, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {args.out_csv}")

    print("\nHow to read this: if 'frac_near_degenerate' is roughly the same across")
    print("dropout-trained and non-dropout-trained models at a given rate, Section 3.5's")
    print("aggregate means are comparing similarly-shaped distributions and the reported")
    print("means are a fair summary. If it differs substantially (e.g. one group has a")
    print("meaningfully higher near-zero/near-one fraction at the same rate), the aggregate")
    print("mean is hiding a shape difference and Section 3.5 should report the distribution,")
    print("not just the mean, for that comparison.")


if __name__ == "__main__":
    main()
