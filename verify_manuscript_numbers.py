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


def verify_table_3_7_claim_a(mosei_raw: pd.DataFrame, v: Verifier):
    """Section 3.7 Claim A: MOSEI graded-robustness significance.

    Recomputed directly from raw per-seed data (mosei_graded_robustness_raw.csv,
    columns: model, seed, missingness_rate, accuracy, f1, n_test, best_val_f1),
    exactly as verify_table_3_2 does for the MOSI comparison — this is the
    stronger from-source check, not a match against an already-aggregated file.
    """
    print("\n--- Table 3.7 Claim A: MOSEI graded-robustness significance (from raw) ---")

    def ttest_and_means(model_a, model_b, rate):
        a = mosei_raw[(mosei_raw.model == model_a) & (mosei_raw.missingness_rate == rate)] \
            .sort_values('seed')['accuracy'].to_numpy()
        b = mosei_raw[(mosei_raw.model == model_b) & (mosei_raw.missingness_rate == rate)] \
            .sort_values('seed')['accuracy'].to_numpy()
        _, p = stats.ttest_rel(a, b)
        return a.mean(), b.mean(), p

    claims = [
        (0.00, 0.7255, 0.7291, 0.424),
        (0.25, 0.6847, 0.6757, 0.298),
        (0.50, 0.6413, 0.6313, 0.308),
        (0.75, 0.6147, 0.5994, 0.189),
    ]
    for rate, mean_a_c, mean_b_c, p_c in claims:
        mean_a, mean_b, p = ttest_and_means("gating_only_no_dropout", "attention_gated_fusion_full", rate)
        v.check(f"MOSEI mean_a @ rate={rate}", mean_a_c, mean_a, tol=0.001)
        v.check(f"MOSEI mean_b @ rate={rate}", mean_b_c, mean_b, tol=0.001)
        v.check(f"MOSEI p_value @ rate={rate}", p_c, p, tol=0.001)


def verify_table_3_7_claim_b(mosei_sm: pd.DataFrame, v: Verifier):
    """Section 3.7 Claim B: MOSEI collapse rate by model.

    Recomputed directly from raw per-seed single-modality results
    (mosei_single_modality_results.csv, columns: model, seed, missing_modality,
    accuracy, f1, n) — this script applies the F1<0.05 collapse threshold
    itself, rather than trusting it was applied correctly upstream in an
    already-aggregated file.
    """
    print("\n--- Table 3.7 Claim B: MOSEI collapse rate by model (from raw) ---")
    text_missing = mosei_sm[mosei_sm.missing_modality == "text"]

    claimed = {
        "attention_gated_fusion_full": 0.0,
        "dropout_only_fusion": 0.0,
        "gating_only_no_dropout": 0.0,
        "hard_mask_gated_fusion": 0.2,
        "fixed_weight_fusion": 0.2,
        "imputation_baseline_post2023": 0.2,
        "early_fusion": 0.4,
        "late_fusion": 0.6,
    }
    for model, rate_c in claimed.items():
        rows = text_missing[text_missing.model == model]
        n_seeds = len(rows)
        n_collapsed = (rows["f1"] < 0.05).sum()
        actual_rate = n_collapsed / n_seeds if n_seeds else float("nan")
        v.check(f"MOSEI collapse rate {model} (n_seeds={n_seeds})", rate_c, actual_rate, tol=0.001)

    # Base-rate cross-check: collapsed (F1<0.05) runs' accuracy should cluster
    # near MOSEI's actual majority-class rate (0.5098), not MOSI's 0.596 —
    # the specific substitution MOSEI_PROVENANCE.md was written to prevent.
    collapsed_rows = text_missing[text_missing["f1"] < 0.05]
    if len(collapsed_rows):
        acc_min, acc_max = collapsed_rows["accuracy"].min(), collapsed_rows["accuracy"].max()
        print(f"  [check] Collapsed-run accuracy range: [{acc_min:.4f}, {acc_max:.4f}] "
              f"— should cluster near MOSEI majority-class rate 0.5098, not MOSI's 0.596.")
        v.check("MOSEI collapsed-run accuracy near base rate (not MOSI's 0.596)",
                 0.5098, (acc_min + acc_max) / 2, tol=0.03)


def verify_section_3_7_1(trainsize_df: pd.DataFrame, baserate_df: pd.DataFrame, v: Verifier):
    """Section 3.7.1: controlled ablation testing the two candidate factors
    for the gating_only_no_dropout MOSEI/MOSI divergence."""
    print("\n--- Section 3.7.1: divergence-factor ablation ---")
    claimed_trainsize_f1 = {42: 0.0406, 123: 0.0546, 2024: 0.1662, 7: 0.6434, 99: 0.0676}
    claimed_baserate_f1 = {42: 0.2725, 123: 0.5610, 2024: 0.4096, 7: 0.5166, 99: 0.4041}

    for seed, f1_c in claimed_trainsize_f1.items():
        row = trainsize_df[trainsize_df.seed == seed]
        if row.empty:
            v.check(f"trainsize ablation row missing for seed={seed}", 1, 0)
            continue
        v.check(f"trainsize f1 @ seed={seed}", f1_c, row.iloc[0]["text_missing_f1"], tol=0.001)

    for seed, f1_c in claimed_baserate_f1.items():
        row = baserate_df[baserate_df.seed == seed]
        if row.empty:
            v.check(f"baserate ablation row missing for seed={seed}", 1, 0)
            continue
        v.check(f"baserate f1 @ seed={seed}", f1_c, row.iloc[0]["text_missing_f1"], tol=0.001)

    n_collapsed_trainsize = (trainsize_df["text_missing_f1"] < 0.05).sum()
    n_collapsed_baserate = (baserate_df["text_missing_f1"] < 0.05).sum()
    v.check("trainsize ablation collapse count", 1, n_collapsed_trainsize)
    v.check("baserate ablation collapse count", 0, n_collapsed_baserate)
    v.check("baserate achieved majority rate (seed=42)", 0.596,
             baserate_df[baserate_df.seed == 42].iloc[0]["achieved_majority_rate"], tol=0.001)


def verify_section_7_item_4(narrow_df: pd.DataFrame, v: Verifier):
    """Section 7 item 4 / mechanism-status table: narrow-dropout-range run.
    Checks the claimed minimum F1 across all 60 rows (3 models x 5 seeds x
    4 rates) matches the manuscript's stated 0.615 floor."""
    print("\n--- Section 7 item 4: narrow-dropout-range run ---")
    v.check("narrow-dropout-range row count", 60, len(narrow_df))
    v.check("narrow-dropout-range min F1 (no collapse anywhere)", 0.615193,
             narrow_df["f1"].min(), tol=0.001)
    v.check("narrow-dropout-range: zero rows below F1<0.05 threshold", 0,
             (narrow_df["f1"] < 0.05).sum())


def verify_section_3_7_1_multidraw(multidraw_df: pd.DataFrame, v: Verifier):
    """Section 3.7.1 follow-up: 25-run multi-draw ablation (5 seeds x 5 draws)."""
    print("\n--- Section 3.7.1 multi-draw follow-up ---")
    v.check("multidraw row count", 25, len(multidraw_df))
    claimed_per_seed = {42: 4, 99: 2, 7: 0, 123: 0, 2024: 0}
    actual_per_seed = multidraw_df.groupby("seed")["collapsed"].sum()
    for seed, claimed in claimed_per_seed.items():
        actual = actual_per_seed.get(seed, None)
        v.check(f"multidraw collapses for seed={seed} (of 5 draws)", claimed, actual)
    v.check("multidraw total collapses", 6, multidraw_df["collapsed"].sum())


def verify_mosi_gating_collapse_labels(single_modality_csv: str, v: Verifier):
    """Sanity check on the CORRECTED CMU-MOSI collapse labels for
    gating_only_no_dropout (Section 3.4/3.6 correction: collapsing seeds are
    42, 123, 2024; non-collapsing are 7, 99 — an earlier revision had these
    inverted in the Section 3.6/4 prose, though Table 3.4's own reported
    3/5=60% rate was always correct)."""
    print("\n--- CMU-MOSI gating_only_no_dropout collapse-label sanity check ---")
    try:
        sm = pd.read_csv(single_modality_csv)
    except FileNotFoundError:
        print(f"  [skip] {single_modality_csv} not found locally.")
        return
    text_missing = sm[(sm.model == "gating_only_no_dropout") & (sm.missing_modality == "text")]
    collapsed_seeds = set(text_missing[text_missing.f1 < 0.05]["seed"])
    expected = {42, 123, 2024}
    ok = collapsed_seeds == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] gating_only_no_dropout MOSI collapsing seeds: "
          f"claimed={expected}  actual={collapsed_seeds}")
    if ok:
        v.passed += 1
    else:
        v.failed += 1
        v.failures.append(f"MOSI collapse-label check: claimed={expected} actual={collapsed_seeds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results_raw.csv")
    parser.add_argument("--config_log", default="config_log.json")
    parser.add_argument("--single_modality_csv", default="single_modality_results.csv")
    parser.add_argument("--single_modality_5seed_csv", default="diagnostics_5seed/single_modality_results.csv")
    parser.add_argument("--gate_weights_summary_csv", default="gate_weights_summary.csv")
    parser.add_argument("--gate_weights_raw_csv", default="gate_weights_raw.csv")
    parser.add_argument("--mosei_raw_csv", default="mosei_graded_robustness_raw.csv")
    parser.add_argument("--mosei_single_modality_csv", default="mosei_single_modality_results.csv")
    parser.add_argument("--mosei_divergence_trainsize_csv", default="mosei_divergence_ablation_trainsize.csv")
    parser.add_argument("--mosei_divergence_baserate_csv", default="mosei_divergence_ablation_baserate.csv")
    parser.add_argument("--narrow_dropout_csv", default="results_narrow_dropout_range.csv")
    parser.add_argument("--mosei_divergence_multidraw_csv", default="mosei_divergence_ablation_trainsize_multidraw.csv")
    parser.add_argument("--skip_mosei", action="store_true",
                         help="Skip Section 3.7 checks (e.g. if the MOSEI raw CSVs aren't present).")
    parser.add_argument("--skip_extras", action="store_true",
                         help="Skip Section 3.7.1 and Section 7 item 4 checks.")
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

    if not args.skip_mosei:
        mosei_raw = pd.read_csv(args.mosei_raw_csv)
        mosei_sm = pd.read_csv(args.mosei_single_modality_csv)
        verify_table_3_7_claim_a(mosei_raw, v)
        verify_table_3_7_claim_b(mosei_sm, v)

    if not args.skip_extras:
        trainsize_df = pd.read_csv(args.mosei_divergence_trainsize_csv)
        baserate_df = pd.read_csv(args.mosei_divergence_baserate_csv)
        verify_section_3_7_1(trainsize_df, baserate_df, v)
        narrow_df = pd.read_csv(args.narrow_dropout_csv)
        verify_section_7_item_4(narrow_df, v)
        multidraw_df = pd.read_csv(args.mosei_divergence_multidraw_csv)
        verify_section_3_7_1_multidraw(multidraw_df, v)
        verify_mosi_gating_collapse_labels(args.single_modality_5seed_csv, v)

    ok = v.summary()
    sys.exit(0 if ok else 1)
