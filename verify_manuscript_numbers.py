"""
verify_manuscript_numbers.py — structural safeguard against unverified/fabricated
numbers making it into the manuscript.

WHAT THIS DOES: recomputes every numeric table currently in the manuscript
(Sections 3.1-3.5) directly from the raw data files, and reports PASS/FAIL
for each claim. It does NOT parse the manuscript's markdown text automatically
(that would require the manuscript's tables to stay in a fixed machine-readable
format, which is more brittle than the value it adds at this project's size) —
instead, each claim is hard-coded here as `(claimed_value, tolerance)` next
to the code that recomputes it from source, so a mismatch is a genuine content
bug in this script or the manuscript, not a parsing artifact.

WHAT THIS DOES NOT DO: verify that a *manuscript claim not represented here*
is real. If a new number is added to the manuscript, add a corresponding
check here in the same edit — this script's coverage is only as good as
whether that discipline is followed. It caught nothing on its own; it only
catches drift between "what this script currently checks" and "what the
manuscript currently claims," so it must be extended alongside the manuscript,
not run once and trusted forever.

USAGE:
    python verify_manuscript_numbers.py \
        --results_csv results_raw.csv \
        --config_log config_log.json \
        --single_modality_csv single_modality_results.csv \
        --gate_weights_summary_csv gate_weights_summary.csv \
        --gate_weights_raw_csv gate_weights_raw.csv

Exits with code 1 if any check fails, so it can be wired into a pre-commit
hook or CI step, not just run manually before sharing a draft.
"""

from __future__ import annotations
import argparse
import json
import sys

import pandas as pd
from scipy import stats


class Verifier:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, label: str, claimed, actual, tol=0.0015):
        ok = abs(claimed - actual) <= tol
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"{label}: claimed={claimed} actual={actual:.5f} (tol={tol})")
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: claimed={claimed}  actual={actual:.5f}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"TOTAL: {self.passed} passed, {self.failed} failed")
        if self.failures:
            print("\nFAILURES:")
            for f in self.failures:
                print(f"  - {f}")
        print(f"{'='*60}")
        return self.failed == 0


def verify_table_3_1(df: pd.DataFrame, v: Verifier):
    """Table 3.1 main means + 95% CI half-widths."""
    print("\n--- Table 3.1: main metric table + CIs ---")
    claimed_means = {
        ('attention_gated_fusion_full', 0.0): 0.764, ('attention_gated_fusion_full', 0.25): 0.686,
        ('attention_gated_fusion_full', 0.5): 0.606, ('attention_gated_fusion_full', 0.75): 0.557,
        ('hard_mask_gated_fusion', 0.0): 0.778, ('hard_mask_gated_fusion', 0.25): 0.699,
        ('hard_mask_gated_fusion', 0.5): 0.621, ('hard_mask_gated_fusion', 0.75): 0.563,
        ('dropout_only_fusion', 0.0): 0.768, ('dropout_only_fusion', 0.25): 0.687,
        ('dropout_only_fusion', 0.5): 0.610, ('dropout_only_fusion', 0.75): 0.558,
        ('gating_only_no_dropout', 0.0): 0.759, ('gating_only_no_dropout', 0.25): 0.713,
        ('gating_only_no_dropout', 0.5): 0.680, ('gating_only_no_dropout', 0.75): 0.653,
        ('fixed_weight_fusion', 0.0): 0.735, ('fixed_weight_fusion', 0.25): 0.684,
        ('fixed_weight_fusion', 0.5): 0.639, ('fixed_weight_fusion', 0.75): 0.618,
        ('early_fusion', 0.0): 0.779, ('early_fusion', 0.25): 0.720,
        ('early_fusion', 0.5): 0.672, ('early_fusion', 0.75): 0.635,
        ('late_fusion', 0.0): 0.779, ('late_fusion', 0.25): 0.713,
        ('late_fusion', 0.5): 0.656, ('late_fusion', 0.75): 0.632,
        ('imputation_baseline_post2023', 0.0): 0.777, ('imputation_baseline_post2023', 0.25): 0.719,
        ('imputation_baseline_post2023', 0.5): 0.679, ('imputation_baseline_post2023', 0.75): 0.653,
    }
    for (model, rate), claimed in claimed_means.items():
        actual = df[(df.model == model) & (df.missingness_rate == rate)]['accuracy'].mean()
        v.check(f"mean acc {model}@{rate}", claimed, actual)

    t_crit = stats.t.ppf(0.975, df=2)
    claimed_ci = {
        ('attention_gated_fusion_full', 0.0): 0.012, ('attention_gated_fusion_full', 0.25): 0.021,
        ('attention_gated_fusion_full', 0.5): 0.024, ('attention_gated_fusion_full', 0.75): 0.026,
        ('dropout_only_fusion', 0.0): 0.037, ('dropout_only_fusion', 0.25): 0.047,
        ('dropout_only_fusion', 0.5): 0.027, ('dropout_only_fusion', 0.75): 0.027,
        ('gating_only_no_dropout', 0.0): 0.015, ('gating_only_no_dropout', 0.25): 0.023,
        ('gating_only_no_dropout', 0.5): 0.046, ('gating_only_no_dropout', 0.75): 0.025,
        ('fixed_weight_fusion', 0.0): 0.089, ('fixed_weight_fusion', 0.25): 0.062,
        ('fixed_weight_fusion', 0.5): 0.108, ('fixed_weight_fusion', 0.75): 0.097,
        ('early_fusion', 0.0): 0.016, ('early_fusion', 0.25): 0.060,
        ('early_fusion', 0.5): 0.094, ('early_fusion', 0.75): 0.104,
        ('late_fusion', 0.0): 0.024, ('late_fusion', 0.25): 0.048,
        ('late_fusion', 0.5): 0.121, ('late_fusion', 0.75): 0.170,
        ('imputation_baseline_post2023', 0.0): 0.016, ('imputation_baseline_post2023', 0.25): 0.064,
        ('imputation_baseline_post2023', 0.5): 0.098, ('imputation_baseline_post2023', 0.75): 0.047,
    }
    for (model, rate), claimed_hw in claimed_ci.items():
        vals = df[(df.model == model) & (df.missingness_rate == rate)]['accuracy'].to_numpy()
        hw = t_crit * vals.std(ddof=1) / (len(vals) ** 0.5)
        v.check(f"CI half-width {model}@{rate}", claimed_hw, hw)


def verify_table_3_2(df: pd.DataFrame, v: Verifier):
    """Section 3.2 paired t-test p-values for every comparison reported."""
    print("\n--- Section 3.2: significance tests ---")

    def ttest(model_a, model_b, rate):
        a = df[(df.model == model_a) & (df.missingness_rate == rate)].sort_values('seed')['accuracy'].to_numpy()
        b = df[(df.model == model_b) & (df.missingness_rate == rate)].sort_values('seed')['accuracy'].to_numpy()
        _, p = stats.ttest_rel(a, b)
        return p

    claims = [
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.0, 0.212),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.25, 0.031),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.5, 0.044),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.75, 0.006),
        ("attention_gated_fusion_full", "dropout_only_fusion", 0.0, 0.786),
        ("attention_gated_fusion_full", "dropout_only_fusion", 0.25, 0.910),
        ("attention_gated_fusion_full", "dropout_only_fusion", 0.5, 0.715),
        ("attention_gated_fusion_full", "dropout_only_fusion", 0.75, 0.423),
        ("gating_only_no_dropout", "imputation_baseline_post2023", 0.0, 0.046),
        ("gating_only_no_dropout", "imputation_baseline_post2023", 0.25, 0.609),
        ("gating_only_no_dropout", "imputation_baseline_post2023", 0.5, 0.982),
        ("gating_only_no_dropout", "imputation_baseline_post2023", 0.75, 0.977),
        ("gating_only_no_dropout", "fixed_weight_fusion", 0.0, 0.303),
        ("gating_only_no_dropout", "fixed_weight_fusion", 0.25, 0.118),
        ("gating_only_no_dropout", "fixed_weight_fusion", 0.5, 0.126),
        ("gating_only_no_dropout", "fixed_weight_fusion", 0.75, 0.335),
        ("hard_mask_gated_fusion", "attention_gated_fusion_full", 0.0, 0.160),
        ("hard_mask_gated_fusion", "attention_gated_fusion_full", 0.25, 0.184),
        ("hard_mask_gated_fusion", "attention_gated_fusion_full", 0.5, 0.109),
        ("hard_mask_gated_fusion", "attention_gated_fusion_full", 0.75, 0.339),
        ("hard_mask_gated_fusion", "gating_only_no_dropout", 0.0, 0.173),
        ("hard_mask_gated_fusion", "gating_only_no_dropout", 0.25, 0.320),
        ("hard_mask_gated_fusion", "gating_only_no_dropout", 0.5, 0.102),
        ("hard_mask_gated_fusion", "gating_only_no_dropout", 0.75, 0.001),
        ("hard_mask_gated_fusion", "dropout_only_fusion", 0.0, 0.597),
        ("hard_mask_gated_fusion", "dropout_only_fusion", 0.25, 0.482),
        ("hard_mask_gated_fusion", "dropout_only_fusion", 0.5, 0.537),
        ("hard_mask_gated_fusion", "dropout_only_fusion", 0.75, 0.469),
    ]
    for model_a, model_b, rate, claimed_p in claims:
        actual_p = ttest(model_a, model_b, rate)
        v.check(f"p-value {model_a} vs {model_b} @{rate}", claimed_p, actual_p, tol=0.001)


def verify_table_3_3(config_log: dict, v: Verifier):
    """Section 3.3 parameter counts."""
    print("\n--- Table 3.3: parameter counts ---")
    claimed = {
        'early_fusion': 503681, 'late_fusion': 454659, 'fixed_weight_fusion': 470913,
        'dropout_only_fusion': 503681, 'gating_only_no_dropout': 495940,
        'attention_gated_fusion_full': 495940, 'hard_mask_gated_fusion': 495940,
        'imputation_baseline_post2023': 701697,
    }
    seen = {}
    for run in config_log['runs']:
        seen.setdefault(run['model'], run['n_params'])
    for model, claimed_params in claimed.items():
        v.check(f"params {model}", claimed_params, seen.get(model, -1), tol=0.5)

    # This check exists because an earlier version of this script's coverage stopped at
    # "does each raw param count match" and missed a real error: the manuscript's prose
    # claimed a %-increase RANGE ("+39-54%, smallest vs X, largest vs Y") where the range
    # bounds were numerically correct but the model LABELS attached to each bound were
    # swapped/wrong. An independent human verifier caught this; a range-bounds-only check
    # would not have. This checks the full (label, percentage) pair, not just the numbers.
    print("\n--- Section 3.3 prose: min/max %-increase claim, including WHICH MODEL label ---")
    imputation_params = seen['imputation_baseline_post2023']
    pct_increases = {m: (imputation_params - p) / p * 100 for m, p in seen.items() if m != 'imputation_baseline_post2023'}
    actual_min_model = min(pct_increases, key=pct_increases.get)
    actual_max_model = max(pct_increases, key=pct_increases.get)
    claimed_min_model = 'early_fusion'  # ties with dropout_only_fusion at same param count
    claimed_max_model = 'late_fusion'
    v.check("min %% increase value", 39.31, pct_increases[actual_min_model], tol=0.1)
    v.check("max %% increase value", 54.33, pct_increases[actual_max_model], tol=0.1)
    label_ok = (pct_increases[claimed_min_model] == pct_increases[actual_min_model]) and (actual_max_model == claimed_max_model)
    if label_ok:
        v.passed += 1
        print(f"[PASS] min/max labels: min-label={claimed_min_model} (actual min={actual_min_model}, tied), max-label={claimed_max_model} matches actual={actual_max_model}")
    else:
        v.failed += 1
        msg = f"min/max labels: claimed min-model={claimed_min_model}, max-model={claimed_max_model} vs. actual min-model={actual_min_model}, actual max-model={actual_max_model}"
        v.failures.append(msg)
        print(f"[FAIL] {msg}")


def verify_table_3_4(sm_df: pd.DataFrame, v: Verifier):
    """Section 3.4 single-modality masking table."""
    print("\n--- Table 3.4: single-modality masking ---")
    claimed = {
        ('gating_only_no_dropout', 'none'): 0.759, ('gating_only_no_dropout', 'text'): 0.596,
        ('gating_only_no_dropout', 'audio'): 0.760, ('gating_only_no_dropout', 'vision'): 0.760,
        ('attention_gated_fusion_full', 'none'): 0.764, ('attention_gated_fusion_full', 'text'): 0.405,
        ('attention_gated_fusion_full', 'audio'): 0.764, ('attention_gated_fusion_full', 'vision'): 0.764,
    }
    for (model, cond), claimed_val in claimed.items():
        actual = sm_df[(sm_df.model == model) & (sm_df.missing_modality == cond)]['accuracy'].mean()
        v.check(f"single-modality {model}/{cond}", claimed_val, actual)


def verify_table_3_5(gw_summary: pd.DataFrame, v: Verifier):
    """Section 3.5 gate weight tables."""
    print("\n--- Table 3.5: gate weight vs. mask sensitivity ---")
    text_claims = {
        (0.25, 'gating_only_no_dropout'): (0.926, 0.761), (0.25, 'attention_gated_fusion_full'): (0.857, 0.692),
        (0.5, 'gating_only_no_dropout'): (0.923, 0.753), (0.5, 'attention_gated_fusion_full'): (0.854, 0.691),
        (0.75, 'gating_only_no_dropout'): (0.923, 0.746), (0.75, 'attention_gated_fusion_full'): (0.846, 0.690),
    }
    for (rate, model), (pres_c, abs_c) in text_claims.items():
        pres = gw_summary[(gw_summary.model == model) & (gw_summary.rate == rate) &
                           (gw_summary.modality == 'text') & (gw_summary.status == 'present')]['mean_weight'].values[0]
        abse = gw_summary[(gw_summary.model == model) & (gw_summary.rate == rate) &
                           (gw_summary.modality == 'text') & (gw_summary.status == 'absent')]['mean_weight'].values[0]
        v.check(f"w(text) present {model}@{rate}", pres_c, pres)
        v.check(f"w(text) absent {model}@{rate}", abs_c, abse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results_raw.csv")
    parser.add_argument("--config_log", default="config_log.json")
    parser.add_argument("--single_modality_csv", default="single_modality_results.csv")
    parser.add_argument("--gate_weights_summary_csv", default="gate_weights_summary.csv")
    parser.add_argument("--gate_weights_raw_csv", default="gate_weights_raw.csv")
    # parse_known_args (not parse_args) so this survives being run inside a Jupyter/Colab
    # cell, which injects its own "-f kernel.json" argument that plain parse_args()
    # rejects as unrecognized. Any genuinely unknown args are silently ignored here,
    # which is a reasonable tradeoff for a script meant to run in notebook cells too.
    args, _unknown = parser.parse_known_args()

    v = Verifier()

    df = pd.read_csv(args.results_csv)
    with open(args.config_log) as f:
        config_log = json.load(f)
    sm_df = pd.read_csv(args.single_modality_csv)
    gw_summary = pd.read_csv(args.gate_weights_summary_csv)

    verify_table_3_1(df, v)
    verify_table_3_2(df, v)
    verify_table_3_3(config_log, v)
    verify_table_3_4(sm_df, v)
    verify_table_3_5(gw_summary, v)

    ok = v.summary()
    sys.exit(0 if ok else 1)
