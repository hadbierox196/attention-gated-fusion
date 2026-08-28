# Attention-Gated Fusion for Modality-Robust Affective Computing — Code + Manuscript

**Status: 5-seed replication complete. The paper's headline finding changed as a result — read `manuscript.md`, not just this file, before trusting any number below.**

This project went through three seed counts and two different headline claims:

1. **3 seeds (42, 123, 2024).** Suggested modality-dropout training hurt an attention gate's missingness robustness, significant at p<0.05 for every nonzero missingness rate.
2. **5 seeds (+ 7, 99).** That result did not replicate — none of the three nonzero-rate comparisons remain significant. See `manuscript.md` Section 3.2 for the full before/after table.
3. **Diagnosing why seeds disagreed** led to a different, better-supported finding: dropout-trained models never collapse to a degenerate constant-output prediction under complete text loss (0/15 seed-runs); non-dropout-trained models do, in a seed-dependent 40–80% of runs. This — not the original ablation — is the paper's current central claim. See `manuscript.md` Sections 3.4 and 4.

If you only read one file before using this repo, read `manuscript.md`, specifically Sections 1.1, 3.2, and 3.4 — they explain what changed and why, and reading only this README will leave you with the *wrong* headline claim (see the incident described below).

## A documentation-staleness incident, kept here deliberately

An earlier version of this README described checkpoints and diagnostics as "not yet executed" for an extended period after they actually had been executed and verified — the README was simply never updated after the underlying work was done. This was not caught by inspection; it was caught by directly probing the filesystem and cross-checking against `verify_manuscript_numbers.py`, and it nearly caused a full, unnecessary re-run of already-completed work. Separately, this project's working copy was at one point missing `data/dataset.py` and `data/__init__.py` entirely (present only in a different archive of the code, never merged in) — meaning a directory that looked complete was not actually runnable. Both incidents are documented in full in `PROVENANCE.md` (Round 4). Treat any status claim in this README, including the one at the top of this file, as something to verify against `verify_manuscript_v2.py`'s actual output before relying on it for anything that matters.

**A related repo-hygiene issue found and fixed while assembling this version:** the working directory previously contained two conflicting `results_raw.csv`-like files with different schemas and different numbers (`results_raw.csv` — the real, current-schema file matching `config_log.json` — and a stray `results_raw (2).csv` with an older 5-column schema and no `n_test`/`best_val_f1` tracking, apparently a leftover from before the Round 1 correction in `PROVENANCE.md`). The stray file has been removed from this version. If you find it again anywhere (another Drive copy, another branch), delete it rather than assuming it's redundant — verify against `verify_manuscript_numbers.py` first if you're not sure which copy is current.

## Structure

```
.
├── README.md                        # this file
├── manuscript.md                    # THE current manuscript — read this for the real findings
├── PROVENANCE.md                    # full correction history, now including Round 4 (this revision)
├── requirements.txt
├── config.yaml                      # seeds: [42, 123, 2024] by default — see note below
├── config_log.json                  # provenance for the original 3-seed run (21 logged runs)
├── results_raw.csv                  # 3-seed results (verified; matches config_log.json's schema)
├── data/
│   ├── __init__.py
│   └── dataset.py                    # CMU-MOSI aligned_50.pkl loading + preprocessing (Section 2.2)
├── models/
│   ├── encoders.py                   # shared GRU modality encoders (Section 2.3)
│   ├── fusion.py                     # all fusion variants (Section 2.3 baseline table)
│   ├── gate.py                       # attention-gated fusion module (Section 2.3)
│   └── hard_mask_gate.py             # hard-mask gate variant
├── missingness.py                    # apply_missingness(batch, rate) (Section 2.4)
├── train.py                          # single (model, seed) training run
├── run_experiment_grid.py            # drives the full grid, writes results_raw.csv, config_log.json, checkpoints/
├── diagnostics.py                    # single-modality masking eval + gate-weight logging
├── run_diagnostics.py                # driver: loads ALL checkpoints in a dir, runs diagnostics.py
├── evaluate.py                       # significance tests (Section 3.2), efficiency table (Section 3.3)
├── sanity_check.py                   # NEW — training-pipeline overfit-a-batch check (Section 2.6)
├── encoder_freeze_swap.py            # NEW — encoder/gate swap test across seeds (Section 3.6 groundwork)
├── run_narrow_dropout_range.py       # NEW — Uniform(0,0.4) training-range variant (Section 4/7)
├── verify_manuscript_numbers.py      # checks the ORIGINAL 3-seed claims (Sections 3.1-3.5, 119 checks)
├── verify_manuscript_v2.py           # NEW — checks the 5-seed/degenerate-collapse/Section 3.6 claims (185 checks)
└── notebooks/
    └── attention_gated_fusion.ipynb  # not audited as part of this revision — verify before trusting
```

## What's included now vs. still only in Drive

As of this revision, the 5-seed replication's **result files** are included directly in this zip: `results_raw_5seed.csv`, `diagnostics_5seed/single_modality_results.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, `encoder_zeroing_isolated_effect.csv`. All five were independently re-verified against `verify_manuscript_v2.py` at the time they were added (185/185 passing, run directly against these exact files, not just spot-checked).

**Still not included, and still only in the Google Drive project folder** (`attention_gated_fusion_project/`) — these are large or environment-specific artifacts not suited to a source zip:

- `data/aligned_50.pkl` (the CMU-MOSI feature data itself)
- `checkpoints/*.pt` (40 files: 8 models × 5 seeds)
- `config_log_5seed.json` (the 5-seed run's provenance log — the *results* made it into this zip, the *log* did not; regenerate or copy over if you need it)
- `significance_5seed_full.csv` (the full 28-row comparison table; `verify_manuscript_v2.py` recomputes the subset it checks directly from `results_raw_5seed.csv`, so this specific file isn't required to verify the manuscript's claims, only to see every comparison at a glance)
- `gate_weights_summary.csv`, `gate_weights_raw.csv` (5-seed versions — the *single-modality* diagnostics file made it in; the full gate-weight-vs-mask analysis behind Section 3.5, which manuscript Section 5 already flags as not yet re-audited, did not)

If you're picking this project back up from scratch, the Drive folder is still the other half of it for anything involving raw checkpoints or the full gate-weight logs — but the numbers needed to verify or extend the manuscript's current claims are now in this zip.

## Run order

**3-seed baseline (original, already done — `results_raw.csv`/`config_log.json` in this repo):**
1. `run_experiment_grid.py` — trains 8 models × 3 seeds.
2. `evaluate.py` — significance tests / CIs.
3. `python verify_manuscript_numbers.py` — should show 119/119 passing against `results_raw.csv`.

**5-seed replication (done, but the outputs live in Drive, not this zip — see above):**
4. `sanity_check.py` — run before trusting a full retrain; see manuscript Section 2.6 for the caveat about dropout-trained models not cleanly passing this check.
5. Retrain only the 2 new seeds (edit `config.yaml`'s `seeds:` field), merge with the existing 3-seed `results_raw.csv`.
6. `run_diagnostics.py --checkpoint_dir <dir with all 5 seeds of checkpoints>` — globs every `*.pt` file it finds, so this naturally covers all 5 seeds once trained.
7. `encoder_freeze_swap.py` and `run_narrow_dropout_range.py` — the two new diagnostic experiments behind manuscript Sections 3.6 and 4/7.
8. `python verify_manuscript_v2.py` — should show 185/185 passing against the 5-seed/diagnostic files listed above.

## What this repo deliberately does not include

- The real CMU-MOSI data (`data/dataset.py` expects `aligned_50.pkl` at a path you provide; see manuscript Section 9).
- Pre-computed 5-seed results (see "What's NOT in this zip" above).
- A working `imputation_baseline_post2023` reconstruction sub-network beyond a minimal stub — see manuscript Section 2.3 for what this stands in for and doesn't reproduce.
