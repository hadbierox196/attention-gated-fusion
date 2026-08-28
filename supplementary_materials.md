# Supplementary Materials

## "Modality Dropout Prevents Degenerate Collapse but Not Graded Missingness Robustness in Attention-Gated Multimodal Fusion: A Replication Study on CMU-MOSI"

**This file was substantially updated alongside the manuscript's Round 4 revision (see `PROVENANCE.md`).** Sections S1-S6 below are the ORIGINAL 3-seed supplementary material, kept for historical transparency exactly as they were first written -- they describe the original ablation claim, which did not replicate at 5 seeds (manuscript Section 3.2) and should not be read as the paper's current finding. Sections S7 onward are new: the 5-seed replication, the corrected single-modality table with F1 (which surfaces the degenerate-collapse artifact undetected in S1-S6), and the mask-channel isolation experiments.

**If you are here to understand what this paper currently claims, start at S7, not S1.** S1-S6 are retained because the numbers in them are real and independently verified (not because they represent the current headline result) -- the same reasoning `manuscript.md` gives for keeping the 3-seed Table 3.1a/3.2a alongside the 5-seed tables rather than deleting them.

All values in S1-S6 are drawn from the original 3-seed `results_raw.csv`, `config_log.json`, `single_modality_results.csv` (3-seed version), `gate_weights_raw.csv`, and `gate_weights_summary.csv`. All values in S7 onward are drawn from `results_raw_5seed.csv`, `diagnostics_5seed/single_modality_results.csv` (5-seed version), `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, and `encoder_zeroing_isolated_effect.csv`. Every table in both halves of this file is independently checkable: S1-S6 via `verify_manuscript_numbers.py`, S7 onward via `verify_manuscript_v2.py` (both included in this repository; both currently pass in full -- 119/119 and 185/185 respectively).

---

## S1. Full per-seed accuracy, 3-seed pilot (underlying Table 3.1a) -- HISTORICAL, SUPERSEDED BY S7

Table 3.1a in the main text reports means and 95% CIs across 3 seeds. Individual per-seed values are given here for full transparency. **This table alone does not tell you the 3-seed ablation claim it supports did not replicate -- see S7 for that.**

| Model | Seed | rate=0.00 | rate=0.25 | rate=0.50 | rate=0.75 |
|---|---|---|---|---|---|
| attention_gated_fusion_full | 42 | 0.7609 | 0.6895 | 0.5962 | 0.5598 |
| attention_gated_fusion_full | 123 | 0.7697 | 0.6764 | 0.6064 | 0.5656 |
| attention_gated_fusion_full | 2024 | 0.7624 | 0.6924 | 0.6152 | 0.5452 |
| dropout_only_fusion | 42 | 0.7843 | 0.7055 | 0.6181 | 0.5627 |
| dropout_only_fusion | 123 | 0.7551 | 0.6676 | 0.5977 | 0.5656 |
| dropout_only_fusion | 2024 | 0.7638 | 0.6880 | 0.6137 | 0.5452 |
| early_fusion | 42 | 0.7726 | 0.7332 | 0.7012 | 0.6487 |
| early_fusion | 123 | 0.7857 | 0.6924 | 0.6297 | 0.5875 |
| early_fusion | 2024 | 0.7799 | 0.7347 | 0.6866 | 0.6676 |
| fixed_weight_fusion | 42 | 0.7405 | 0.7099 | 0.6895 | 0.6399 |
| fixed_weight_fusion | 123 | 0.7682 | 0.6822 | 0.6122 | 0.5729 |
| fixed_weight_fusion | 2024 | 0.6968 | 0.6603 | 0.6166 | 0.6414 |
| gating_only_no_dropout | 42 | 0.7609 | 0.7230 | 0.6997 | 0.6429 |
| gating_only_no_dropout | 123 | 0.7638 | 0.7055 | 0.6764 | 0.6633 |
| gating_only_no_dropout | 2024 | 0.7522 | 0.7099 | 0.6633 | 0.6531 |
| hard_mask_gated_fusion | 42 | 0.7624 | 0.6895 | 0.6006 | 0.5583 |
| hard_mask_gated_fusion | 123 | 0.7872 | 0.6968 | 0.6283 | 0.5700 |
| hard_mask_gated_fusion | 2024 | 0.7843 | 0.7114 | 0.6327 | 0.5598 |
| imputation_baseline_post2023 | 42 | 0.7843 | 0.7420 | 0.7172 | 0.6545 |
| imputation_baseline_post2023 | 123 | 0.7741 | 0.6910 | 0.6385 | 0.6327 |
| imputation_baseline_post2023 | 2024 | 0.7726 | 0.7245 | 0.6822 | 0.6706 |
| late_fusion | 42 | 0.7682 | 0.6924 | 0.6006 | 0.5525 |
| late_fusion | 123 | 0.7843 | 0.7172 | 0.6880 | 0.6720 |
| late_fusion | 2024 | 0.7857 | 0.7303 | 0.6808 | 0.6706 |

---

## S2. Full gate weight summary, 3-seed pilot (underlying Table 3.5) -- NOT YET RE-AUDITED FOR THE DEGENERATE-COLLAPSE ARTIFACT

Main text Section 3.5 reports `gating_only_no_dropout` and `attention_gated_fusion_full` only, for the text modality. Full table below includes `hard_mask_gated_fusion` and all three modalities.

**Caution carried over from `manuscript.md` Section 5:** this table's "absent" rows are aggregate means over per-sample gate weights. S8 below shows that a meaningful fraction of individual predictions in the analogous single-modality accuracy table were degenerate constant-output collapses, invisible in an accuracy-only view. This table has not been re-checked for whether a similar per-sample artifact distorts these aggregate weight means -- listed as Future Work item 3 in the manuscript. Treat the numbers below as unaudited in that specific sense, even though they are correctly computed from real data.

| Model | Rate | Modality | Status | Mean weight | Std | n |
|---|---|---|---|---|---|---|
| attention_gated_fusion_full | 0.25 | audio | absent | 0.0853 | 0.0732 | 492 |
| attention_gated_fusion_full | 0.25 | audio | present | 0.0820 | 0.0714 | 1566 |
| attention_gated_fusion_full | 0.25 | text | absent | 0.6918 | 0.1621 | 487 |
| attention_gated_fusion_full | 0.25 | text | present | 0.8569 | 0.1413 | 1571 |
| attention_gated_fusion_full | 0.25 | vision | absent | 0.1011 | 0.0909 | 513 |
| attention_gated_fusion_full | 0.25 | vision | present | 0.0988 | 0.0928 | 1545 |
| attention_gated_fusion_full | 0.50 | audio | absent | 0.0932 | 0.0736 | 968 |
| attention_gated_fusion_full | 0.50 | audio | present | 0.1041 | 0.0756 | 1090 |
| attention_gated_fusion_full | 0.50 | text | absent | 0.6906 | 0.1625 | 905 |
| attention_gated_fusion_full | 0.50 | text | present | 0.8543 | 0.1419 | 1153 |
| attention_gated_fusion_full | 0.50 | vision | absent | 0.1123 | 0.0979 | 939 |
| attention_gated_fusion_full | 0.50 | vision | present | 0.1240 | 0.0988 | 1119 |
| attention_gated_fusion_full | 0.75 | audio | absent | 0.1041 | 0.0766 | 1224 |
| attention_gated_fusion_full | 0.75 | audio | present | 0.1290 | 0.0713 | 834 |
| attention_gated_fusion_full | 0.75 | text | absent | 0.6904 | 0.1650 | 1272 |
| attention_gated_fusion_full | 0.75 | text | present | 0.8460 | 0.1490 | 786 |
| attention_gated_fusion_full | 0.75 | vision | absent | 0.1226 | 0.1010 | 1284 |
| attention_gated_fusion_full | 0.75 | vision | present | 0.1581 | 0.1006 | 774 |
| gating_only_no_dropout | 0.25 | audio | absent | 0.0510 | 0.0651 | 492 |
| gating_only_no_dropout | 0.25 | audio | present | 0.0521 | 0.0668 | 1566 |
| gating_only_no_dropout | 0.25 | text | absent | 0.7608 | 0.1761 | 487 |
| gating_only_no_dropout | 0.25 | text | present | 0.9264 | 0.0834 | 1571 |
| gating_only_no_dropout | 0.25 | vision | absent | 0.0595 | 0.0719 | 513 |
| gating_only_no_dropout | 0.25 | vision | present | 0.0614 | 0.0727 | 1545 |
| gating_only_no_dropout | 0.50 | audio | absent | 0.0640 | 0.0766 | 968 |
| gating_only_no_dropout | 0.50 | audio | present | 0.0734 | 0.0803 | 1090 |
| gating_only_no_dropout | 0.50 | text | absent | 0.7533 | 0.1777 | 905 |
| gating_only_no_dropout | 0.50 | text | present | 0.9235 | 0.0856 | 1153 |
| gating_only_no_dropout | 0.50 | vision | absent | 0.0754 | 0.0823 | 939 |
| gating_only_no_dropout | 0.50 | vision | present | 0.0882 | 0.0897 | 1119 |
| gating_only_no_dropout | 0.75 | audio | absent | 0.0699 | 0.0791 | 1224 |
| gating_only_no_dropout | 0.75 | audio | present | 0.1061 | 0.0909 | 834 |
| gating_only_no_dropout | 0.75 | text | absent | 0.7456 | 0.1779 | 1272 |
| gating_only_no_dropout | 0.75 | text | present | 0.9229 | 0.0873 | 786 |
| gating_only_no_dropout | 0.75 | vision | absent | 0.0908 | 0.0902 | 1284 |
| gating_only_no_dropout | 0.75 | vision | present | 0.1210 | 0.0986 | 774 |
| hard_mask_gated_fusion | 0.25 | audio | absent | 0.0000 | 0.0000 | 492 |
| hard_mask_gated_fusion | 0.25 | audio | present | 0.1560 | 0.2844 | 1566 |
| hard_mask_gated_fusion | 0.25 | text | absent | 0.0000 | 0.0000 | 487 |
| hard_mask_gated_fusion | 0.25 | text | present | 0.9835 | 0.0150 | 1571 |
| hard_mask_gated_fusion | 0.25 | vision | absent | 0.0000 | 0.0000 | 513 |
| hard_mask_gated_fusion | 0.25 | vision | present | 0.1738 | 0.3025 | 1545 |
| hard_mask_gated_fusion | 0.50 | audio | absent | 0.0000 | 0.0000 | 968 |
| hard_mask_gated_fusion | 0.50 | audio | present | 0.4010 | 0.4255 | 1090 |
| hard_mask_gated_fusion | 0.50 | text | absent | 0.0000 | 0.0000 | 905 |
| hard_mask_gated_fusion | 0.50 | text | present | 0.9893 | 0.0135 | 1153 |
| hard_mask_gated_fusion | 0.50 | vision | absent | 0.0000 | 0.0000 | 939 |
| hard_mask_gated_fusion | 0.50 | vision | present | 0.4291 | 0.4318 | 1119 |
| hard_mask_gated_fusion | 0.75 | audio | absent | 0.0000 | 0.0000 | 1224 |
| hard_mask_gated_fusion | 0.75 | audio | present | 0.7893 | 0.3699 | 834 |
| hard_mask_gated_fusion | 0.75 | text | absent | 0.0000 | 0.0000 | 1272 |
| hard_mask_gated_fusion | 0.75 | text | present | 0.9963 | 0.0090 | 786 |
| hard_mask_gated_fusion | 0.75 | vision | absent | 0.0000 | 0.0000 | 1284 |
| hard_mask_gated_fusion | 0.75 | vision | present | 0.7967 | 0.3605 | 774 |

Note: `hard_mask_gated_fusion`'s "absent" rows are exactly 0.0000 +/- 0.0000 by construction (Section 2.3, `HardMaskAttentionGate`), not an empirical result -- the gate structurally cannot assign weight to a masked-out modality. This is included as a confirmation that the implementation matches its intended design, not as a finding.

---

## S3. Distributional detail on the gate-weight "flattening" effect, 3-seed pilot (Section 3.5, main text)

Main text Section 3.5 reports mean w(text) when text is present; this shows the full distribution isn't concentrated at the mean but shifted throughout. Same caveat as S2 applies: not yet re-audited against the degenerate-collapse finding in S8 below.

**Percentiles of w(text) when text is present** (pooled across all three eval rates):

| Percentile | gating_only_no_dropout | attention_gated_fusion_full |
|---|---|---|
| min | 0.500 | 0.226 |
| 10th | 0.799 | 0.657 |
| 25th | 0.881 | 0.787 |
| 50th (median) | 0.949 | 0.876 |
| 75th | 0.999 | 0.982 |
| 90th | 1.000 | 0.992 |
| max | 1.000 | 0.998 |
| mean | 0.925 | 0.854 |
| n | 3510 | 3510 |

**Vision-over-audio fallback preference**, all three gated models, when text is absent and both audio and vision are present:

| Model | w(audio) | w(vision) | vision/audio ratio | n |
|---|---|---|---|---|
| gating_only_no_dropout | 0.1058 | 0.1323 | 1.251 | 637 |
| attention_gated_fusion_full | 0.1379 | 0.1690 | 1.225 | 637 |
| hard_mask_gated_fusion | 0.4525 | 0.5475 | 1.210 | 637 |

This ratio (vision favored ~21-25% over audio) is stable across all three training regimes, suggesting it reflects a genuine property of the audio vs. vision encoders' relative informativeness on CMU-MOSI rather than anything related to the dropout-training question investigated in the main text. Noted here for completeness; not further interpreted.

---

## S4. Full training configuration, 3-seed pilot (from `config_log.json`)

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 15 |
| Hidden dim (all encoders + gate) | 128 |
| Seeds | 42, 123, 2024 |
| Training-time missingness rate | Uniform(0, 0.75), resampled per batch |
| Evaluation missingness rates | 0.00, 0.25, 0.50, 0.75 |
| `fixed_weight_fusion` prior | (0.4, 0.3, 0.3) for (text, audio, vision) |
| Models trained | early_fusion, late_fusion, fixed_weight_fusion, dropout_only_fusion, gating_only_no_dropout, imputation_baseline_post2023, attention_gated_fusion_full, hard_mask_gated_fusion |

## S5. Software environment, 3-seed pilot run (from `config_log.json`)

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.35 |
| Device | CUDA (Google Colab T4 GPU session) |
| Run timestamp (UTC) | 2026-08-26T08:50:52 |

## S6. Reproducibility (3-seed pilot)

All data in S1-S5 is generated by `run_experiment_grid.py` and `run_diagnostics.py`. As of the Round 4 revision, the previously-cited Zenodo DOI (`10.5281/zenodo.22105162`) has not been independently confirmed to resolve -- see `manuscript.md` Section 9 and `README.md`. Every table in S1-S5 can be independently recomputed and checked against `results_raw.csv`, `config_log.json`, and the 3-seed diagnostics CSVs using `verify_manuscript_numbers.py` (119/119 passing as of this revision).

---

## S7. Full per-seed accuracy, 5-seed replication (underlying Table 3.1b) -- CURRENT

Same structure as S1, extended to 5 seeds (adds 7, 99). This is the data underlying the manuscript's current Table 3.1b and the non-replication reported in Section 3.2.


| Model | Seed | rate=0.00 | rate=0.25 | rate=0.50 | rate=0.75 |
|---|---|---|---|---|---|
| attention_gated_fusion_full | 42 | 0.7609 | 0.6895 | 0.5962 | 0.5598 |
| attention_gated_fusion_full | 123 | 0.7697 | 0.6764 | 0.6064 | 0.5656 |
| attention_gated_fusion_full | 2024 | 0.7624 | 0.6924 | 0.6152 | 0.5452 |
| attention_gated_fusion_full | 7 | 0.7697 | 0.6910 | 0.5948 | 0.5452 |
| attention_gated_fusion_full | 99 | 0.7609 | 0.6822 | 0.5948 | 0.5554 |
| hard_mask_gated_fusion | 42 | 0.7624 | 0.6895 | 0.6006 | 0.5583 |
| hard_mask_gated_fusion | 123 | 0.7872 | 0.6968 | 0.6283 | 0.5700 |
| hard_mask_gated_fusion | 2024 | 0.7843 | 0.7114 | 0.6327 | 0.5598 |
| hard_mask_gated_fusion | 7 | 0.7449 | 0.6647 | 0.5802 | 0.5364 |
| hard_mask_gated_fusion | 99 | 0.7828 | 0.6953 | 0.6079 | 0.5685 |
| dropout_only_fusion | 42 | 0.7843 | 0.7055 | 0.6181 | 0.5627 |
| dropout_only_fusion | 123 | 0.7551 | 0.6676 | 0.5977 | 0.5656 |
| dropout_only_fusion | 2024 | 0.7638 | 0.6880 | 0.6137 | 0.5452 |
| dropout_only_fusion | 7 | 0.7770 | 0.6939 | 0.5977 | 0.5569 |
| dropout_only_fusion | 99 | 0.7872 | 0.7012 | 0.6093 | 0.5641 |
| gating_only_no_dropout | 42 | 0.7609 | 0.7230 | 0.6997 | 0.6429 |
| gating_only_no_dropout | 123 | 0.7638 | 0.7055 | 0.6764 | 0.6633 |
| gating_only_no_dropout | 2024 | 0.7522 | 0.7099 | 0.6633 | 0.6531 |
| gating_only_no_dropout | 7 | 0.7609 | 0.6778 | 0.5904 | 0.5510 |
| gating_only_no_dropout | 99 | 0.7493 | 0.6706 | 0.5948 | 0.5481 |
| fixed_weight_fusion | 42 | 0.7405 | 0.7099 | 0.6895 | 0.6399 |
| fixed_weight_fusion | 123 | 0.7682 | 0.6822 | 0.6122 | 0.5729 |
| fixed_weight_fusion | 2024 | 0.6968 | 0.6603 | 0.6166 | 0.6414 |
| fixed_weight_fusion | 7 | 0.7595 | 0.7318 | 0.6720 | 0.6501 |
| fixed_weight_fusion | 99 | 0.7711 | 0.7303 | 0.6997 | 0.6706 |
| early_fusion | 42 | 0.7726 | 0.7332 | 0.7012 | 0.6487 |
| early_fusion | 123 | 0.7857 | 0.6924 | 0.6297 | 0.5875 |
| early_fusion | 2024 | 0.7799 | 0.7347 | 0.6866 | 0.6676 |
| early_fusion | 7 | 0.7653 | 0.7420 | 0.6793 | 0.6516 |
| early_fusion | 99 | 0.7682 | 0.6808 | 0.5977 | 0.5496 |
| late_fusion | 42 | 0.7682 | 0.6924 | 0.6006 | 0.5525 |
| late_fusion | 123 | 0.7843 | 0.7172 | 0.6880 | 0.6720 |
| late_fusion | 2024 | 0.7857 | 0.7303 | 0.6808 | 0.6706 |
| late_fusion | 7 | 0.7682 | 0.6793 | 0.5918 | 0.5466 |
| late_fusion | 99 | 0.7843 | 0.6997 | 0.6064 | 0.5598 |
| imputation_baseline_post2023 | 42 | 0.7843 | 0.7420 | 0.7172 | 0.6545 |
| imputation_baseline_post2023 | 123 | 0.7741 | 0.6910 | 0.6385 | 0.6327 |
| imputation_baseline_post2023 | 2024 | 0.7726 | 0.7245 | 0.6822 | 0.6706 |
| imputation_baseline_post2023 | 7 | 0.7668 | 0.7434 | 0.6764 | 0.6589 |
| imputation_baseline_post2023 | 99 | 0.7566 | 0.7128 | 0.6910 | 0.6764 |

---

## S8. Full single-modality masking table with F1, all 8 models, all 5 seeds (underlying Table 3.4) -- CURRENT, THE CORRECTED FINDING

This is the corrected version of the original single-modality table, now including F1 alongside accuracy. Cells where accuracy = 0.596210 (= 409/686, this test set's negative-class base rate) with F1 near 0.000 are degenerate constant-output collapses, not evidence of robustness -- see `manuscript.md` Section 3.4 for the full explanation. `--` denotes a cell not independently confirmed for the manuscript's Table 3.4 (only `late_fusion`, seed 123 was confirmed there; the table below reports every cell present in the underlying CSV regardless).

| Model | Seed | Condition | Accuracy | F1 | Degenerate? |
|---|---|---|---|---|---|
| attention_gated_fusion_full | 7 | text-missing | 0.4038 | 0.5753 |  |
| attention_gated_fusion_full | 42 | text-missing | 0.4067 | 0.5756 |  |
| attention_gated_fusion_full | 99 | text-missing | 0.4052 | 0.5750 |  |
| attention_gated_fusion_full | 123 | text-missing | 0.4038 | 0.5753 |  |
| attention_gated_fusion_full | 2024 | text-missing | 0.4052 | 0.5750 |  |
| hard_mask_gated_fusion | 7 | text-missing | 0.4023 | 0.5738 |  |
| hard_mask_gated_fusion | 42 | text-missing | 0.4067 | 0.5756 |  |
| hard_mask_gated_fusion | 99 | text-missing | 0.4082 | 0.5753 |  |
| hard_mask_gated_fusion | 123 | text-missing | 0.4038 | 0.5753 |  |
| hard_mask_gated_fusion | 2024 | text-missing | 0.4111 | 0.5765 |  |
| dropout_only_fusion | 7 | text-missing | 0.4052 | 0.5750 |  |
| dropout_only_fusion | 42 | text-missing | 0.4096 | 0.5759 |  |
| dropout_only_fusion | 99 | text-missing | 0.4038 | 0.5753 |  |
| dropout_only_fusion | 123 | text-missing | 0.4038 | 0.5753 |  |
| dropout_only_fusion | 2024 | text-missing | 0.4067 | 0.5765 |  |
| gating_only_no_dropout | 7 | text-missing | 0.4038 | 0.5753 |  |
| gating_only_no_dropout | 42 | text-missing | 0.5962 | 0.0000 | YES |
| gating_only_no_dropout | 99 | text-missing | 0.4038 | 0.5753 |  |
| gating_only_no_dropout | 123 | text-missing | 0.5962 | 0.0000 | YES |
| gating_only_no_dropout | 2024 | text-missing | 0.5962 | 0.0000 | YES |
| fixed_weight_fusion | 7 | text-missing | 0.5962 | 0.0000 | YES |
| fixed_weight_fusion | 42 | text-missing | 0.5962 | 0.0000 | YES |
| fixed_weight_fusion | 99 | text-missing | 0.5889 | 0.0208 |  |
| fixed_weight_fusion | 123 | text-missing | 0.4096 | 0.5768 |  |
| fixed_weight_fusion | 2024 | text-missing | 0.5962 | 0.0000 | YES |
| early_fusion | 7 | text-missing | 0.5962 | 0.0000 | YES |
| early_fusion | 42 | text-missing | 0.5962 | 0.0000 | YES |
| early_fusion | 99 | text-missing | 0.4038 | 0.5753 |  |
| early_fusion | 123 | text-missing | 0.4155 | 0.5783 |  |
| early_fusion | 2024 | text-missing | 0.5962 | 0.0000 | YES |
| late_fusion | 7 | text-missing | 0.4038 | 0.5753 |  |
| late_fusion | 42 | text-missing | 0.4038 | 0.5753 |  |
| late_fusion | 99 | text-missing | 0.4038 | 0.5753 |  |
| late_fusion | 123 | text-missing | 0.5962 | 0.0000 | YES |
| late_fusion | 2024 | text-missing | 0.5962 | 0.0000 | YES |
| imputation_baseline_post2023 | 7 | text-missing | 0.5962 | 0.0072 | YES |
| imputation_baseline_post2023 | 42 | text-missing | 0.5962 | 0.0000 | YES |
| imputation_baseline_post2023 | 99 | text-missing | 0.5918 | 0.0210 |  |
| imputation_baseline_post2023 | 123 | text-missing | 0.4125 | 0.5771 |  |
| imputation_baseline_post2023 | 2024 | text-missing | 0.5962 | 0.0000 | YES |

**Summary:** 14 of 40 text-missing seed-runs across all 8 models are degenerate (F1<0.01). Restricted to the 3 dropout-trained models: 0 of 15 are degenerate. This 0-vs-nonzero split, exactly reproducing across every seed tested, is the paper's current central finding.

---

## S9. Mask-channel and encoder-zeroing isolation experiments (underlying Section 3.6) -- CURRENT, NEW

Full per-seed data behind the two isolation experiments in `manuscript.md` Section 3.6, run on `gating_only_no_dropout` checkpoints across all 5 seeds. Not yet run on `attention_gated_fusion_full` -- listed as Future Work item 2.

**Mask-channel isolation** (real, non-zeroed text features; only the mask value fed to the gate changes):

| Seed | w(text), mask=on | w(text), mask=off | Shift |
|---|---|---|---|

| 7 | 0.7232 | 0.7136 | -0.0096 |
| 42 | 0.8602 | 0.8511 | -0.0091 |
| 99 | 0.8878 | 0.8807 | -0.0071 |
| 123 | 0.9993 | 0.9993 | -0.0001 |
| 2024 | 0.9286 | 0.9236 | -0.0050 |

**Encoder-zeroing isolation** (mask channel held at [1,1,1]; only the encoder's text input is zeroed):

| Seed | w(text), real input | w(text), zeroed input | Shift |
|---|---|---|---|

| 7 | 0.7232 | 0.5465 | -0.1767 |
| 42 | 0.8602 | 0.6148 | -0.2454 |
| 99 | 0.8878 | 0.6179 | -0.2699 |
| 123 | 0.9993 | 0.9970 | -0.0024 |
| 2024 | 0.9286 | 0.7076 | -0.2210 |

**Gate first-layer weight norms, feature block vs. mask block** (underlying the architectural claim in `manuscript.md` Section 2.3 that `gating_only_no_dropout`'s mask-input weights never received a varying training signal):

| Seed | Group | Feature-block norm | Mask-block norm |
|---|---|---|---|

| 7 | collapsed(text-missing~0.40) | 4.9474 | 0.4782 |
| 42 | non-collapsed(~0.596) | 5.0249 | 0.4311 |
| 99 | collapsed(text-missing~0.40) | 5.2329 | 0.4964 |
| 123 | non-collapsed(~0.596) | 5.3142 | 0.5067 |
| 2024 | non-collapsed(~0.596) | 5.3724 | 0.5288 |

---

## S10. Reproducibility (5-seed replication and diagnostics)

S7-S9 above are generated by `run_experiment_grid.py` (2 additional seeds trained into the same checkpoint directory as the 3-seed run), `run_diagnostics.py` (re-run against the full 5-seed checkpoint set), and `encoder_freeze_swap.py`/a standalone mask-channel isolation script (both included in this repository). Every table in S7-S9 is independently checkable against `results_raw_5seed.csv`, `diagnostics_5seed/single_modality_results.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, and `encoder_zeroing_isolated_effect.csv` using `verify_manuscript_v2.py` (185/185 passing as of this revision, confirmed directly against these exact files).

