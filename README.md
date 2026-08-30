# Attention-Gated Fusion for Modality-Robust Affective Computing — Code + Manuscript

**Status: 5-seed CMU-MOSI replication complete, plus a scoped CMU-MOSEI cross-dataset extension (also complete). The paper's headline finding changed twice as a result — read `manuscript.md`, not just this file, before trusting any number below.**

This project went through three seed counts, two different headline claims, and one cross-dataset extension:

1. **3 seeds (42, 123, 2024), CMU-MOSI only.** Suggested modality-dropout training hurt an attention gate's missingness robustness, significant at p<0.05 for every nonzero missingness rate.
2. **5 seeds (+ 7, 99), CMU-MOSI only.** That result did not replicate — none of the three nonzero-rate comparisons remain significant. See `manuscript.md` Section 3.2 for the full before/after table.
3. **Diagnosing why seeds disagreed** led to a different, better-supported finding: dropout-trained models never collapse to a degenerate constant-output prediction under complete text loss (0/15 seed-runs); non-dropout-trained models do, in a seed-dependent 40–80% of runs on CMU-MOSI. This — not the original ablation — is the paper's central claim. See `manuscript.md` Sections 3.4 and 4.
4. **A scoped, two-claim extension to CMU-MOSEI** (8 models x 5 seeds, the same two claims above) mostly confirms this: the graded-robustness non-effect replicates cleanly, and dropout-trained models again show zero collapses. But one non-dropout model (`gating_only_no_dropout`) shows zero collapses on MOSEI too — a genuine divergence from the MOSI pattern, reported directly rather than smoothed over. See `manuscript.md` Section 3.7 and `supplementary_materials.md` S11 for the full per-seed data.

If you only read one file before using this repo, read `manuscript.md`, specifically Sections 1.1, 3.2, 3.4, and 3.7 — they explain what changed and why, and reading only this README will leave you with an incomplete picture of the headline claims (see the incident described below).

## A documentation-staleness incident, kept here deliberately

An earlier version of this README described checkpoints and diagnostics as "not yet executed" for an extended period after they actually had been executed and verified — the README was simply never updated after the underlying work was done. This was not caught by inspection; it was caught by directly probing the filesystem and cross-checking against `verify_manuscript_numbers.py`, and it nearly caused a full, unnecessary re-run of already-completed work. Separately, this project's working copy was at one point missing `data/dataset.py` and `data/__init__.py` entirely (present only in a different archive of the code, never merged in) — meaning a directory that looked complete was not actually runnable. A closely related version of the same incident happened again during the CMU-MOSEI extension: repeated Colab RAM-crash recoveries silently reset `config.yaml`'s `mosei:` block, `data/dataset_mosei.py`'s expected-dimension constants, and `run_mosei_targeted_replication.py`'s checkpointing logic back to their stale pre-patch state on every re-extraction from the same zip, because that zip only ever contained the original unpatched files. All of these incidents are documented in full in `PROVENANCE.md`. Treat any status claim in this README, including the one at the top of this file, as something to verify against `verify_manuscript_numbers.py`'s actual output before relying on it for anything that matters.

**A related repo-hygiene issue found and fixed while assembling this version:** the working directory previously contained two conflicting `results_raw.csv`-like files with different schemas and different numbers (`results_raw.csv` — the real, current-schema file matching `config_log.json` — and a stray `results_raw (2).csv` with an older 5-column schema and no `n_test`/`best_val_f1` tracking, apparently a leftover from before the Round 1 correction in `PROVENANCE.md`). The stray file has been removed from this version. If you find it again anywhere (another Drive copy, another branch), delete it rather than assuming it's redundant — verify against `verify_manuscript_numbers.py` first if you're not sure which copy is current.

## Structure

```
.
├── README.md                        # this file
├── manuscript.md                    # THE current manuscript — read this for the real findings
├── supplementary_materials.md       # S1-S10: CMU-MOSI (3-seed + 5-seed); S11: CMU-MOSEI (new)
├── PROVENANCE.md                    # full correction history (5 rounds, including this revision)
├── MOSEI_PROVENANCE.md              # feature-provenance sign-off for the CMU-MOSEI extension
├── requirements.txt
├── config.yaml                      # top-level seeds: [42,123,2024,7,99] for CMU-MOSI; separate mosei: block for CMU-MOSEI
├── config_log.json                  # provenance for the original 3-seed CMU-MOSI run
├── results_raw.csv                  # 3-seed CMU-MOSI results (verified; matches config_log.json)
├── results_raw_5seed.csv            # 5-seed CMU-MOSI results (verified via verify_manuscript_v2.py)
├── data/
│   ├── __init__.py
│   ├── dataset.py                    # CMU-MOSI aligned_50.pkl loading + preprocessing (Section 2.2)
│   └── dataset_mosei.py              # CMU-MOSEI loader — module-level tensor cache (RAM-crash fix,
│                                      #   see PROVENANCE.md), mandatory FEATURE_RELEASE_NOTE guard
├── models/
│   ├── encoders.py                   # shared GRU modality encoders (Section 2.3)
│   ├── fusion.py                     # all fusion variants (Section 2.3 baseline table)
│   ├── gate.py                       # attention-gated fusion module (Section 2.3)
│   └── hard_mask_gate.py             # hard-mask gate variant
├── missingness.py                    # apply_missingness(batch, rate) (Section 2.4)
├── train.py                          # single (model, seed) training run
├── run_experiment_grid.py            # drives the full CMU-MOSI grid, writes results_raw.csv
├── run_mosei_targeted_replication.py # drives the scoped 2-claim CMU-MOSEI run (Section 3.7);
│                                      #   self-contained with incremental CSV checkpointing,
│                                      #   resume-from-CSV support, and timing/ETA instrumentation
├── mosei_graded_robustness_raw.csv   # real per-seed CMU-MOSEI results, Claim A (Section 3.7)
├── mosei_single_modality_results.csv # real per-seed CMU-MOSEI results, Claim B (Section 3.7)
├── diagnostics.py                    # single-modality masking eval + gate-weight logging
├── run_diagnostics.py                # driver: loads ALL checkpoints in a dir, runs diagnostics.py
├── evaluate.py                       # significance tests (Section 3.2), efficiency table (Section 3.3)
├── sanity_check.py                   # training-pipeline overfit-a-batch check (Section 2.6);
│                                      #   --dataset {mosi,mosei} flag supports both datasets
├── encoder_freeze_swap.py            # encoder/gate swap test across seeds (Section 3.6 groundwork)
├── run_narrow_dropout_range.py       # Uniform(0,0.4) training-range variant (Section 4/7)
├── verify_manuscript_numbers.py      # checks Sections 3.1-3.5 (119 checks) AND Section 3.7 (21 checks,
│                                      #   recomputed from raw per-seed MOSEI data, not just matched
│                                      #   against an aggregate file — see --skip_mosei to disable)
├── verify_manuscript_v2.py           # checks the 5-seed/degenerate-collapse/Section 3.6 claims (185 checks)
├── CITATION.cff                      # needs real author/ORCID/DOI info filled in — see TODOs in the file
├── mask_channel_isolation.py         # Section 7 item 3 — mask-channel isolation on attention_gated_fusion_full;
│                                      #   run this revision, real result in mask_channel_isolated_effect_dropout_trained.csv
├── run_mosei_divergence_ablation.py  # Section 3.7.1 — controlled test of the 2 candidate factors for the
│                                      #   gating_only_no_dropout MOSEI/MOSI divergence; run this revision
├── reaudit_gate_weights_per_sample.py # Section 3.5 re-audit against the Section 3.4 collapse artifact;
│                                      #   run this revision, real result in gate_weights_per_sample_reaudit.csv
├── mask_channel_isolated_effect_dropout_trained.csv  # real result, attention_gated_fusion_full, 5 seeds
├── gate_weights_per_sample_reaudit.csv               # real result, per-sample distributional re-audit
├── mosei_divergence_ablation_trainsize.csv           # real result, Section 3.7.1
├── mosei_divergence_ablation_baserate.csv            # real result, Section 3.7.1
├── results_narrow_dropout_range.csv                  # real result, Section 7 item 4
└── notebooks/
    └── attention_gated_fusion.ipynb  # not audited as part of this revision — verify before trusting
```

## What's included now vs. still only in Drive

As of this revision, the 5-seed CMU-MOSI replication's **result files** are included directly in this zip: `results_raw_5seed.csv`, `diagnostics_5seed/single_modality_results.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, `encoder_zeroing_isolated_effect.csv`. All five were independently re-verified against `verify_manuscript_v2.py` at the time they were added (185/185 passing, run directly against these exact files, not just spot-checked). The full CMU-MOSEI extension's **result files** are also included: `mosei_graded_robustness_raw.csv` and `mosei_single_modality_results.csv`, both real per-seed output from the actual 40-run (8 models x 5 seeds) Colab execution, independently re-verified against `verify_manuscript_numbers.py`'s Section 3.7 checks (21/21 passing, recomputed from these exact files, not matched against a pre-aggregated summary).

**Still not included, and still only in the Google Drive project folder** (`attention_gated_fusion_project/`) — these are large or environment-specific artifacts not suited to a source zip:

- `data/aligned_50.pkl` (CMU-MOSI) and `data/mosei_aligned_50.pkl` (CMU-MOSEI) — the raw feature data itself
- `checkpoints/*.pt` (CMU-MOSI: 40 files, 8 models x 5 seeds after the full replication; CMU-MOSEI: 40 files, 8 models x 5 seeds from the targeted replication)
- `config_log_5seed.json` (the 5-seed CMU-MOSI run's provenance log — the *results* made it into this zip, the *log* did not; regenerate or copy over if you need it)
- `significance_5seed_full.csv` (the full 28-row CMU-MOSI comparison table; `verify_manuscript_v2.py` recomputes the subset it checks directly from `results_raw_5seed.csv`, so this specific file isn't required to verify the manuscript's claims, only to see every comparison at a glance)
- `gate_weights_summary.csv`, `gate_weights_raw.csv` (5-seed CMU-MOSI versions — the *single-modality* diagnostics file made it in; the full gate-weight-vs-mask analysis behind Section 3.5, which manuscript Section 5 already flags as not yet re-audited, did not)

If you're picking this project back up from scratch, the Drive folder is still the other half of it for anything involving raw checkpoints or the full gate-weight logs — but the numbers needed to verify or extend the manuscript's current claims (both datasets) are now in this zip.

**A known gap, flagged rather than hidden:** this repo copy does not currently include the original 3-seed `single_modality_results.csv` or `gate_weights_summary.csv` (the files `verify_manuscript_numbers.py`'s Section 3.4/3.5 checks require by default) — only the 5-seed and CMU-MOSEI equivalents made it into this zip. Running `verify_manuscript_numbers.py` as-is will fail with `FileNotFoundError` on those two arguments until you supply them (from the Drive folder, or by regenerating via `run_diagnostics.py` against the original 3-seed checkpoints). This does not block the CMU-MOSEI checks, which only need the two MOSEI files already in this zip; to run just those, either supply dummy/real values for the other required args, or call `verify_table_3_7_claim_a`/`verify_table_3_7_claim_b` directly as shown in this project's own development process.

## Run order

**CMU-MOSI, 3-seed baseline (original, already done — `results_raw.csv`/`config_log.json` in this repo):**
1. `run_experiment_grid.py` — trains 8 models × 3 seeds.
2. `evaluate.py` — significance tests / CIs.
3. `python verify_manuscript_numbers.py --skip_mosei` — should show 119/119 passing against `results_raw.csv`, once you also supply `single_modality_results.csv` and `gate_weights_summary.csv` (see the known gap noted above; not needed for this step if you're running immediately after step 1-2 in a fresh environment where you generated them yourself).

**CMU-MOSI, 5-seed replication (done; outputs live in this repo — see above):**
4. `sanity_check.py --dataset mosi` — run before trusting a full retrain; see manuscript Section 2.6 for the caveat about dropout-trained models not cleanly passing this check at low step budgets.
5. Retrain only the 2 new seeds (edit `config.yaml`'s top-level `seeds:` field), merge with the existing 3-seed `results_raw.csv`.
6. `run_diagnostics.py --checkpoint_dir <dir with all 5 seeds of checkpoints>` — globs every `*.pt` file it finds, so this naturally covers all 5 seeds once trained.
7. `encoder_freeze_swap.py` and `run_narrow_dropout_range.py` — the two diagnostic experiments behind manuscript Sections 3.6 and 4/7.
8. `python verify_manuscript_v2.py` — should show 185/185 passing against the 5-seed/diagnostic files listed above.

**CMU-MOSEI cross-dataset extension (done; outputs included in this repo — see above):**
9. Fill in `MOSEI_PROVENANCE.md` **before** running anything — confirm feature dimensions and the actual test-set base rate against the real loaded data, do not assume they match CMU-MOSI's. This repo's copy is already filled in with the values actually observed at runtime.
10. `sanity_check.py --dataset mosei` — same caveat as step 4; escalate the step budget if dropout-trained models don't cleanly converge at the default budget (this is expected, not a bug — see manuscript Section 2.6).
11. `run_mosei_targeted_replication.py` — runs exactly the two claims described in manuscript Section 3.7 across 8 models x 5 seeds. Supports resuming from a partially-complete CSV (checks which (seed, model) pairs are already written and skips them), which matters if training is interrupted — see `PROVENANCE.md` for why this was necessary in practice.
12. `python verify_manuscript_numbers.py --mosei_raw_csv mosei_graded_robustness_raw.csv --mosei_single_modality_csv mosei_single_modality_results.csv` — the Section 3.7 + 3.7.1 + Section 7 item 4 portions (37 checks total) pass against the real result files already in this zip; the Section 3.1-3.5 portion (119 checks) additionally needs the two files noted in the known gap above.

## Four analyses completed this revision (Section 3.5, 3.6, 3.7.1, and 7 item 4)

Four open items from an earlier checklist were executed against real data this revision, not just prepared as scripts. All four have real result CSVs in this zip and are independently re-verified by `verify_manuscript_numbers.py` (`--skip_extras` to skip these four specifically, if you only want the core MOSI/MOSEI checks):

- **Section 3.5's aggregate gate-weight means, re-audited per-sample** (`gate_weights_per_sample_reaudit.csv`) — no distortion found in the reported means, but a real distributional difference emerged that the means alone didn't show (`gating_only_no_dropout` has a fatter near-1 tail than `attention_gated_fusion_full` at every rate). `hard_mask_gated_fusion`'s large near-degenerate fraction turned out to be architectural (its masked-softmax structurally forces exact 0/1 weights), not a discovered pathology — worth knowing before reading that row.
- **Section 3.6's mask-channel isolation, extended to `attention_gated_fusion_full`** (`mask_channel_isolated_effect_dropout_trained.csv`) — every shift under 0.012 in magnitude, extending the mask-inertness finding to the dropout-trained gate.
- **Section 3.7.1: controlled test of the two candidate MOSEI/MOSI divergence factors** (`mosei_divergence_ablation_trainsize.csv`, `mosei_divergence_ablation_baserate.csv`) — training-set size is the better-supported factor (1/5 collapses when matched to CMU-MOSI's size, vs. 0/5 for base-rate matching alone), but the evidence is directional, not conclusive. `run_mosei_divergence_ablation.py`'s `baserate` mode had a real bug during this session (it only handled lowering the majority rate, and silently no-op'd when the target was above the natural rate) — fixed in the version now in this repo; the CSV reflects the corrected run.
- **Section 7 item 4: narrow-dropout-range training** (`results_narrow_dropout_range.csv`) — no collapse observed under graded random missingness up to rate=0.75 (min F1=0.615 across all 60 model/seed/rate combinations). Read the caveat in Section 7 item 4 before citing this: it tests Section 3.2's graded random-missingness protocol, not Section 3.4's guaranteed-complete-text-loss protocol, so it's suggestive rather than a direct confirmation.

## Publishing this repo: GitHub → Zenodo

If you're pushing this zip's contents to a GitHub repository to mint a fresh Zenodo DOI (e.g. because the current DOI, `10.5281/zenodo.22141293`, needs re-verification or replacement — see manuscript Section 9 and Future Work item 5):

1. **Do not commit `data/*.pkl`.** These are excluded from this zip on purpose (see "What's included now vs. still only in Drive" above) — they're large and, depending on the CMU-MOSI/CMU-MOSEI license terms, may not be freely redistributable. Add a `.gitignore` entry for `data/*.pkl` and `checkpoints/*.pt` before your first commit, not after.
2. **Fill in `CITATION.cff` and add a `LICENSE` file before your first commit.** `CITATION.cff` is included as a template with explicit `TODO` placeholders (real author name/ORCID/affiliation, license, repository URL) — it does not currently have your real information, and no `LICENSE` file exists in this repo at all as of this revision. Both feed directly into what Zenodo displays as the deposit's author/license metadata once the GitHub integration archives a release, so get them right before the first tagged release rather than after.
3. **Push to GitHub first, tag a release.** `git init`, add everything else in this zip, commit, push, then create a GitHub Release (a tagged version, e.g. `v2.0-mosei-extension`) — Zenodo's GitHub integration archives a specific release, not just the default branch's latest state.
4. **Enable the repo in Zenodo's GitHub integration** (https://zenodo.org/account/settings/github/), flip the toggle for this repository *before* creating the release if you want that specific release auto-archived; if you already created the release, you can still trigger an archive by creating a new release/tag after enabling the toggle.
5. **Zenodo will mint a new DOI automatically** on release creation, tied to that specific tagged version, and will also expose a version-independent "concept DOI" that always resolves to the latest version.
6. **Update the manuscript with the new DOI.** Once you have it, replace `10.5281/zenodo.22141293` throughout `manuscript.md` and `run_experiment_grid.py`/`supplementary_materials.md` (same DOI string appears in a few places — grep for it) — this is the exact update pattern already used once in this project's history (see `PROVENANCE.md`). Also update the `identifiers:` DOI field in `CITATION.cff`.
7. **Verify the new DOI resolves in a logged-out session** before treating it as final, per the standing caveat in manuscript Section 9 — a DOI that only resolves while logged into your own Zenodo account is not yet a usable Data Availability link.

## What this repo deliberately does not include

- The real CMU-MOSI or CMU-MOSEI data (`data/dataset.py`/`data/dataset_mosei.py` expect the respective `.pkl` file at a path you provide; see manuscript Section 9).
- Pre-computed checkpoints for either dataset (see "What's NOT in this zip" above).
- A working `imputation_baseline_post2023` reconstruction sub-network beyond a minimal stub — see manuscript Section 2.3 for what this stands in for and doesn't reproduce.
