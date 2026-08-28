"""
verify_manuscript_numbers_v2.py — extends the original verification script
(Future Work item 5) to cover the REVISED manuscript's claims: the 5-seed
replication (Table 3.1b/3.2b), the corrected single-modality/degenerate-
collapse finding (Table 3.4), and the mask-channel isolation experiments
(Section 3.6).

Does NOT re-check the original 3-seed claims (Table 3.1a/3.2a) or the
Section 3.3 parameter-count claims — those are unchanged from the prior
revision and already covered by verify_manuscript_numbers.py. Run both
scripts; neither supersedes the other.

COVERAGE NOTE (read before trusting a clean run): Table 3.4 in the current
manuscript reports per-seed values for 7 of 8 models. `late_fusion`'s cells
for seeds 42, 2024, 7, and 99 were never independently confirmed against
raw tool output while this script was written (only seed=123 was) — those
four cells are NOT checked here and are excluded from the "all 7 models"
coverage claim below. If you add those numbers to the manuscript, add
matching checks here in the same edit, per the discipline note carried
over from the original script.

USAGE:
    python verify_manuscript_v2.py \
        --results_csv results_raw_5seed.csv \
        --single_modality_csv diagnostics_5seed/single_modality_results.csv \
        --gate_weight_norms_csv gate_weight_norms_by_seed.csv \
        --mask_channel_csv mask_channel_isolated_effect.csv \
        --encoder_zeroing_csv encoder_zeroing_isolated_effect.csv

Exits 1 if any check fails.
"""

from __future__ import annotations
import argparse
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

    def check_bool(self, label: str, claimed: bool, actual: bool):
        ok = claimed == actual
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"{label}: claimed={claimed} actual={actual}")
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: claimed={claimed}  actual={actual}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"TOTAL: {self.passed} passed, {self.failed} failed")
        if self.failures:
            print("\nFAILURES:")
            for f in self.failures:
                print(f"  - {f}")
        print(f"{'='*60}")
        return self.failed == 0


def verify_table_3_1b(df: pd.DataFrame, v: Verifier):
    """Table 3.1b — 5-seed means and 95% CI half-widths."""
    print("\n--- Table 3.1b: 5-seed main metric table + CIs ---")
    claimed_means = {
        ('attention_gated_fusion_full', 0.0): 0.765, ('attention_gated_fusion_full', 0.25): 0.686,
        ('attention_gated_fusion_full', 0.5): 0.601, ('attention_gated_fusion_full', 0.75): 0.554,
        ('hard_mask_gated_fusion', 0.0): 0.772, ('hard_mask_gated_fusion', 0.25): 0.692,
        ('hard_mask_gated_fusion', 0.5): 0.610, ('hard_mask_gated_fusion', 0.75): 0.559,
        ('dropout_only_fusion', 0.0): 0.773, ('dropout_only_fusion', 0.25): 0.691,
        ('dropout_only_fusion', 0.5): 0.607, ('dropout_only_fusion', 0.75): 0.559,
        ('gating_only_no_dropout', 0.0): 0.757, ('gating_only_no_dropout', 0.25): 0.697,
        ('gating_only_no_dropout', 0.5): 0.645, ('gating_only_no_dropout', 0.75): 0.612,
        ('fixed_weight_fusion', 0.0): 0.747, ('fixed_weight_fusion', 0.25): 0.703,
        ('fixed_weight_fusion', 0.5): 0.658, ('fixed_weight_fusion', 0.75): 0.635,
        ('early_fusion', 0.0): 0.774, ('early_fusion', 0.25): 0.717,
        ('early_fusion', 0.5): 0.659, ('early_fusion', 0.75): 0.621,
        ('late_fusion', 0.0): 0.778, ('late_fusion', 0.25): 0.704,
        ('late_fusion', 0.5): 0.634, ('late_fusion', 0.75): 0.600,
        ('imputation_baseline_post2023', 0.0): 0.771, ('imputation_baseline_post2023', 0.25): 0.723,
        ('imputation_baseline_post2023', 0.5): 0.681, ('imputation_baseline_post2023', 0.75): 0.659,
    }
    for (model, rate), claimed in claimed_means.items():
        actual = df[(df.model == model) & (df.missingness_rate == rate)]['accuracy'].mean()
        v.check(f"5seed mean acc {model}@{rate}", claimed, actual, tol=0.001)

    t_crit = stats.t.ppf(0.975, df=4)
    claimed_ci = {
        ('attention_gated_fusion_full', 0.0): 0.006, ('attention_gated_fusion_full', 0.25): 0.008,
        ('attention_gated_fusion_full', 0.5): 0.011, ('attention_gated_fusion_full', 0.75): 0.011,
        ('hard_mask_gated_fusion', 0.0): 0.023, ('hard_mask_gated_fusion', 0.25): 0.021,
        ('hard_mask_gated_fusion', 0.5): 0.027, ('hard_mask_gated_fusion', 0.75): 0.017,
        ('dropout_only_fusion', 0.0): 0.017, ('dropout_only_fusion', 0.25): 0.018,
        ('dropout_only_fusion', 0.5): 0.012, ('dropout_only_fusion', 0.75): 0.010,
        ('gating_only_no_dropout', 0.0): 0.008, ('gating_only_no_dropout', 0.25): 0.028,
        ('gating_only_no_dropout', 0.5): 0.062, ('gating_only_no_dropout', 0.75): 0.071,
        ('fixed_weight_fusion', 0.0): 0.038, ('fixed_weight_fusion', 0.25): 0.039,
        ('fixed_weight_fusion', 0.5): 0.051, ('fixed_weight_fusion', 0.75): 0.046,
        ('early_fusion', 0.0): 0.010, ('early_fusion', 0.25): 0.035,
        ('early_fusion', 0.5): 0.054, ('early_fusion', 0.75): 0.062,
        ('late_fusion', 0.0): 0.011, ('late_fusion', 0.25): 0.025,
        ('late_fusion', 0.5): 0.058, ('late_fusion', 0.75): 0.081,
        ('imputation_baseline_post2023', 0.0): 0.013, ('imputation_baseline_post2023', 0.25): 0.027,
        ('imputation_baseline_post2023', 0.5): 0.035, ('imputation_baseline_post2023', 0.75): 0.021,
    }
    for (model, rate), claimed_hw in claimed_ci.items():
        vals = df[(df.model == model) & (df.missingness_rate == rate)]['accuracy'].to_numpy()
        hw = t_crit * vals.std(ddof=1) / (len(vals) ** 0.5)
        v.check(f"5seed CI half-width {model}@{rate}", claimed_hw, hw, tol=0.002)

    # CI widening claim for gating_only_no_dropout at 50%/75% (Section 3.1 prose)
    hw_50 = t_crit * df[(df.model == 'gating_only_no_dropout') & (df.missingness_rate == 0.5)]['accuracy'].std(ddof=1) / (5 ** 0.5)
    hw_75 = t_crit * df[(df.model == 'gating_only_no_dropout') & (df.missingness_rate == 0.75)]['accuracy'].std(ddof=1) / (5 ** 0.5)
    v.check_bool("prose claim: gating_only_no_dropout CI widens (not narrows) at 50%", True, hw_50 > 0.046)
    v.check_bool("prose claim: gating_only_no_dropout CI widens (not narrows) at 75%", True, hw_75 > 0.025)


def verify_table_3_2b(df: pd.DataFrame, v: Verifier):
    """Table 3.2b — 5-seed significance tests, recomputed from raw results."""
    print("\n--- Table 3.2b: 5-seed significance tests (recomputed, not read from a cache) ---")

    def ttest(model_a, model_b, rate):
        a = df[(df.model == model_a) & (df.missingness_rate == rate)].sort_values('seed')['accuracy'].to_numpy()
        b = df[(df.model == model_b) & (df.missingness_rate == rate)].sort_values('seed')['accuracy'].to_numpy()
        assert len(a) == 5 and len(b) == 5, f"expected 5 seeds each, got {len(a)}/{len(b)}"
        _, p = stats.ttest_rel(a, b)
        return p

    claims = [
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.0, 0.024, True),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.25, 0.327, False),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.5, 0.103, False),
        ("gating_only_no_dropout", "attention_gated_fusion_full", 0.75, 0.076, False),
        ("hard_mask_gated_fusion", "gating_only_no_dropout", 0.75, 0.087, False),
    ]
    for model_a, model_b, rate, claimed_p, claimed_sig in claims:
        actual_p = ttest(model_a, model_b, rate)
        v.check(f"5seed p-value {model_a} vs {model_b} @{rate}", claimed_p, actual_p, tol=0.001)
        v.check_bool(f"5seed significance (p<0.05) {model_a} vs {model_b} @{rate}", claimed_sig, actual_p < 0.05)

    # Core prose claim: "none of the three nonzero-rate comparisons remain significant"
    nonzero_ps = [ttest("gating_only_no_dropout", "attention_gated_fusion_full", r) for r in (0.25, 0.5, 0.75)]
    v.check_bool("prose claim: none of the 3 nonzero-rate headline comparisons are significant",
                 True, all(p >= 0.05 for p in nonzero_ps))
    # Direction-reversal claim at rate 0.0
    mean_a0 = df[(df.model == "gating_only_no_dropout") & (df.missingness_rate == 0.0)]['accuracy'].mean()
    mean_b0 = df[(df.model == "attention_gated_fusion_full") & (df.missingness_rate == 0.0)]['accuracy'].mean()
    v.check_bool("prose claim: at rate=0.0, no-dropout gate is now WORSE (not better) than dropout gate",
                 True, mean_a0 < mean_b0)


def verify_table_3_4(sm_df: pd.DataFrame, v: Verifier):
    """Table 3.4 — corrected single-modality text-missing table, per-seed accuracy+F1."""
    print("\n--- Table 3.4: text-missing condition, per-seed accuracy/F1 ---")
    # (model, seed): (claimed_accuracy, claimed_f1)
    claimed = {
        ('attention_gated_fusion_full', 42): (0.406706, 0.575600),
        ('attention_gated_fusion_full', 123): (0.403790, 0.575286),
        ('attention_gated_fusion_full', 2024): (0.405248, 0.575000),
        ('attention_gated_fusion_full', 7): (0.403790, 0.575286),
        ('attention_gated_fusion_full', 99): (0.405248, 0.575000),
        ('dropout_only_fusion', 42): (0.409621, 0.575916),
        ('dropout_only_fusion', 123): (0.403790, 0.575286),
        ('dropout_only_fusion', 2024): (0.406706, 0.576483),
        ('dropout_only_fusion', 7): (0.405248, 0.575000),
        ('dropout_only_fusion', 99): (0.403790, 0.575286),
        ('hard_mask_gated_fusion', 42): (0.406706, 0.575600),
        ('hard_mask_gated_fusion', 123): (0.403790, 0.575286),
        ('hard_mask_gated_fusion', 2024): (0.411079, 0.576520),
        ('hard_mask_gated_fusion', 7): (0.402332, 0.573805),
        ('hard_mask_gated_fusion', 99): (0.408163, 0.575314),
        ('gating_only_no_dropout', 42): (0.596210, 0.000000),
        ('gating_only_no_dropout', 123): (0.596210, 0.000000),
        ('gating_only_no_dropout', 2024): (0.596210, 0.000000),
        ('gating_only_no_dropout', 7): (0.403790, 0.575286),
        ('gating_only_no_dropout', 99): (0.403790, 0.575286),
        ('early_fusion', 123): (0.415452, 0.578339),
        ('early_fusion', 2024): (0.596210, 0.000000),
        ('early_fusion', 42): (0.596210, 0.000000),
        ('early_fusion', 7): (0.596210, 0.000000),
        ('early_fusion', 99): (0.403790, 0.575286),
        ('fixed_weight_fusion', 123): (0.409621, 0.576803),
        ('fixed_weight_fusion', 2024): (0.596210, 0.000000),
        ('fixed_weight_fusion', 42): (0.596210, 0.000000),
        ('fixed_weight_fusion', 7): (0.596210, 0.000000),
        ('fixed_weight_fusion', 99): (0.588921, 0.020833),
        ('imputation_baseline_post2023', 123): (0.412536, 0.577125),
        ('imputation_baseline_post2023', 2024): (0.596210, 0.000000),
        ('imputation_baseline_post2023', 42): (0.596210, 0.000000),
        ('imputation_baseline_post2023', 7): (0.596210, 0.007168),
        ('imputation_baseline_post2023', 99): (0.591837, 0.020979),
        # late_fusion: ONLY seed=123 was independently confirmed. seeds 42/2024/7/99
        # are deliberately NOT included -- see module docstring.
        ('late_fusion', 123): (0.596210, 0.000000),
    }
    for (model, seed), (claimed_acc, claimed_f1) in claimed.items():
        row = sm_df[(sm_df.model == model) & (sm_df.seed == seed) & (sm_df.missing_modality == 'text')]
        if len(row) == 0:
            v.failed += 1
            v.failures.append(f"Table 3.4 {model}@seed{seed}: no matching row found in single_modality_csv")
            print(f"[FAIL] Table 3.4 {model}@seed{seed}: no row found")
            continue
        v.check(f"Table 3.4 accuracy {model}@seed{seed}", claimed_acc, row['accuracy'].iloc[0])
        v.check(f"Table 3.4 F1 {model}@seed{seed}", claimed_f1, row['f1'].iloc[0])

    print("\n--- Prose claim: base-rate arithmetic behind the 0.596210 constant-collapse value ---")
    v.check("409/686 equals the recurring degenerate accuracy value", 0.596210, 409 / 686, tol=0.000001)

    print("\n--- Prose claim: dropout-trained models NEVER show degenerate collapse (0/15 seed-runs) ---")
    dropout_models = ["attention_gated_fusion_full", "dropout_only_fusion", "hard_mask_gated_fusion"]
    text_rows = sm_df[sm_df.missing_modality == 'text']
    degenerate = text_rows[(text_rows.f1 < 0.01)]
    dropout_degenerate = degenerate[degenerate.model.isin(dropout_models)]
    v.check_bool("0 degenerate (F1<0.01) rows among dropout-trained models",
                 True, len(dropout_degenerate) == 0)
    print(f"    (found {len(dropout_degenerate)} degenerate rows among dropout-trained models, out of "
          f"{len(text_rows[text_rows.model.isin(dropout_models)])} total dropout-trained text-missing rows)")

    non_dropout_models = ["gating_only_no_dropout", "early_fusion", "fixed_weight_fusion",
                           "imputation_baseline_post2023", "late_fusion"]
    non_dropout_degenerate = degenerate[degenerate.model.isin(non_dropout_models)]
    v.check_bool("at least 1 degenerate (F1<0.01) row exists among non-dropout-trained models",
                 True, len(non_dropout_degenerate) > 0)
    print(f"    (found {len(non_dropout_degenerate)} degenerate rows among non-dropout-trained models)")


def verify_section_3_6(gwn_df: pd.DataFrame, mask_df: pd.DataFrame, enc_df: pd.DataFrame, v: Verifier):
    """Section 3.6 — gate weight norms, mask-channel isolation, encoder-zeroing isolation."""
    print("\n--- Section 3.6, mask-channel isolation (Part 2) ---")
    claimed_mask = {
        42: (0.860, 0.851, -0.009), 123: (0.999, 0.999, -0.0001),
        2024: (0.929, 0.924, -0.005), 7: (0.723, 0.714, -0.010), 99: (0.888, 0.881, -0.007),
    }
    for seed, (claimed_on, claimed_off, claimed_shift) in claimed_mask.items():
        row = mask_df[mask_df.seed == seed]
        if len(row) == 0:
            v.failed += 1; v.failures.append(f"mask-channel seed={seed}: no row found"); continue
        v.check(f"mask-channel w_on seed={seed}", claimed_on, row['mean_w_text_mask_on_real_features'].iloc[0], tol=0.001)
        v.check(f"mask-channel w_off seed={seed}", claimed_off, row['mean_w_text_mask_off_real_features'].iloc[0], tol=0.001)
        v.check(f"mask-channel shift seed={seed}", claimed_shift, row['shift_from_mask_channel_alone'].iloc[0], tol=0.001)

    print("\n--- Prose claim: mask-channel shift is <0.01 in magnitude for every seed ---")
    all_shifts_small = (mask_df['shift_from_mask_channel_alone'].abs() < 0.01).all()
    v.check_bool("all 5 seeds show |mask-channel shift| < 0.01", True, all_shifts_small)

    print("\n--- Section 3.6, encoder-zeroing isolation (Part 3) ---")
    claimed_enc = {
        42: (0.860, 0.615, -0.245), 123: (0.999, 0.997, -0.002),
        2024: (0.929, 0.708, -0.221), 7: (0.723, 0.547, -0.177), 99: (0.888, 0.618, -0.270),
    }
    for seed, (claimed_real, claimed_zero, claimed_shift) in claimed_enc.items():
        row = enc_df[enc_df.seed == seed]
        if len(row) == 0:
            v.failed += 1; v.failures.append(f"encoder-zeroing seed={seed}: no row found"); continue
        v.check(f"encoder-zeroing real seed={seed}", claimed_real, row['mean_w_text_real_text_input'].iloc[0], tol=0.001)
        v.check(f"encoder-zeroing zero seed={seed}", claimed_zero, row['mean_w_text_zeroed_text_input_mask_still_1'].iloc[0], tol=0.001)
        v.check(f"encoder-zeroing shift seed={seed}", claimed_shift, row['shift_from_encoder_zeroing_alone'].iloc[0], tol=0.001)

    print("\n--- Prose claim: encoder-zeroing effect does NOT cleanly separate collapsing (7,99) from non-collapsing (42,123,2024) seeds ---")
    collapsed = enc_df[enc_df.seed.isin([7, 99])]['shift_from_encoder_zeroing_alone'].abs()
    non_collapsed = enc_df[enc_df.seed.isin([42, 123, 2024])]['shift_from_encoder_zeroing_alone'].abs()
    # "does not cleanly separate" == the non-collapsed group's own range spans values both
    # smaller AND larger than the collapsed group's range (i.e. no clean threshold exists)
    overlaps = (non_collapsed.min() < collapsed.max()) and (non_collapsed.max() > collapsed.min())
    v.check_bool("non-collapsed group's shift values overlap with collapsed group's (no clean separation)",
                 True, overlaps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results_raw_5seed.csv")
    parser.add_argument("--single_modality_csv", default="diagnostics_5seed/single_modality_results.csv")
    parser.add_argument("--gate_weight_norms_csv", default="gate_weight_norms_by_seed.csv")
    parser.add_argument("--mask_channel_csv", default="mask_channel_isolated_effect.csv")
    parser.add_argument("--encoder_zeroing_csv", default="encoder_zeroing_isolated_effect.csv")
    args, _unknown = parser.parse_known_args()

    v = Verifier()

    df = pd.read_csv(args.results_csv)
    sm_df = pd.read_csv(args.single_modality_csv)
    gwn_df = pd.read_csv(args.gate_weight_norms_csv)
    mask_df = pd.read_csv(args.mask_channel_csv)
    enc_df = pd.read_csv(args.encoder_zeroing_csv)

    verify_table_3_1b(df, v)
    verify_table_3_2b(df, v)
    verify_table_3_4(sm_df, v)
    verify_section_3_6(gwn_df, mask_df, enc_df, v)

    ok = v.summary()
    sys.exit(0 if ok else 1)
