"""
Post-hoc analysis of results_raw.csv — Section 3.2 (significance tests) and
Section 3.3 (efficiency table). Run after run_experiment_grid.py.
"""

from __future__ import annotations
import argparse

import pandas as pd
from scipy import stats


def paired_ttest_by_rate(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """
    Paired t-test (paired on seed) comparing model_a vs model_b accuracy at
    each missingness rate — Section 3.2. With 3 seeds this gives df=2;
    see manuscript Section 5 (Limitations) on statistical power at this n.
    """
    rows = []
    for rate in sorted(df["missingness_rate"].unique()):
        a = (
            df[(df.model == model_a) & (df.missingness_rate == rate)]
            .sort_values("seed")["accuracy"]
            .to_numpy()
        )
        b = (
            df[(df.model == model_b) & (df.missingness_rate == rate)]
            .sort_values("seed")["accuracy"]
            .to_numpy()
        )
        if len(a) != len(b) or len(a) < 2:
            continue
        t_stat, p_val = stats.ttest_rel(a, b)
        rows.append(
            {
                "missingness_rate": rate,
                "model_a": model_a,
                "model_b": model_b,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "diff": a.mean() - b.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "n_seeds": len(a),
            }
        )
    return pd.DataFrame(rows)


def confidence_intervals(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """95% CI per (model, rate) via t-distribution, matching manuscript Table 3.1's CI block."""
    rows = []
    for (model, rate), group in df.groupby(["model", "missingness_rate"]):
        vals = group["accuracy"].to_numpy()
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        if n > 1:
            t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            half_width = t_crit * std / (n ** 0.5)
        else:
            half_width = float("nan")
        rows.append(
            {
                "model": model,
                "missingness_rate": rate,
                "mean": mean,
                "std": std,
                "ci_half_width": half_width,
                "n": n,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "missingness_rate"])


def efficiency_table(config_log_path: str) -> pd.DataFrame:
    """Params per model from config_log.json — Section 3.3."""
    import json

    with open(config_log_path) as f:
        log = json.load(f)
    rows = []
    seen = set()
    for run in log["runs"]:
        if run["model"] in seen:
            continue
        seen.add(run["model"])
        rows.append({"model": run["model"], "n_params": run["n_params"]})
    return pd.DataFrame(rows).sort_values("n_params")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results_raw.csv")
    parser.add_argument("--config_log", default="config_log.json")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    print("=== Confidence intervals (Table 3.1 CI block) ===")
    print(confidence_intervals(df).to_string(index=False))

    print("\n=== Paired t-test: attention_gated_fusion_full vs dropout_only_fusion (Section 3.2) ===")
    print(
        paired_ttest_by_rate(df, "attention_gated_fusion_full", "dropout_only_fusion").to_string(
            index=False
        )
    )

    print("\n=== Efficiency table (Section 3.3) ===")
    print(efficiency_table(args.config_log).to_string(index=False))
