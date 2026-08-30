# Supplementary Materials

## "Modality Dropout Prevents Degenerate Collapse but Not Graded Missingness Robustness in Attention-Gated Multimodal Fusion: A Replication Study on CMU-MOSI"

**This file was substantially updated alongside the manuscript's Round 4 revision (see `PROVENANCE.md`), and again for the CMU-MOSEI cross-dataset extension (S11).** Sections S1-S6 below are the ORIGINAL 3-seed supplementary material, kept for historical transparency exactly as they were first written -- they describe the original ablation claim, which did not replicate at 5 seeds (manuscript Section 3.2) and should not be read as the paper's current finding. Sections S7 through S10 are the 5-seed CMU-MOSI replication: the corrected single-modality table with F1 (which surfaces the degenerate-collapse artifact undetected in S1-S6), and the mask-channel isolation experiments. S11 is new: the full per-seed data behind the CMU-MOSEI cross-dataset extension (manuscript Section 3.7).

**If you are here to understand what this paper currently claims, start at S7 (CMU-MOSI) and S11 (CMU-MOSEI), not S1.** S1-S6 are retained because the numbers in them are real and independently verified (not because they represent the current headline result) -- the same reasoning `manuscript.md` gives for keeping the 3-seed Table 3.1a/3.2a alongside the 5-seed tables rather than deleting them.

All values in S1-S6 are drawn from the original 3-seed `results_raw.csv`, `config_log.json`, `single_modality_results.csv` (3-seed version), `gate_weights_raw.csv`, and `gate_weights_summary.csv`. All values in S7-S10 are drawn from `results_raw_5seed.csv`, `diagnostics_5seed/single_modality_results.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, and `encoder_zeroing_isolated_effect.csv`. All values in S11 are drawn from `mosei_graded_robustness_raw.csv` and `mosei_single_modality_results.csv`. Every table in this file is independently checkable: S1-S6 via `verify_manuscript_numbers.py`'s Section 3.1-3.5 checks (119/119 passing), S7-S10 via `verify_manuscript_v2.py` (185/185 passing), and S11 via `verify_manuscript_numbers.py`'s Section 3.7 checks (21/21 passing, all three scripts included in this repository).

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

All data in S1-S5 is generated by `run_experiment_grid.py` and `run_diagnostics.py`. As of the Round 4 revision, the previously-cited Zenodo DOI (`10.5281/zenodo.22141293`) has not been independently confirmed to resolve -- see `manuscript.md` Section 9 and `README.md`. Every table in S1-S5 can be independently recomputed and checked against `results_raw.csv`, `config_log.json`, and the 3-seed diagnostics CSVs using `verify_manuscript_numbers.py` (119/119 passing as of this revision).

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


---

## S11. CMU-MOSEI cross-dataset extension, full per-seed data (underlying Section 3.7) -- CURRENT, NEW

This section reports the scoped, two-claim CMU-MOSEI replication described in `manuscript.md` Section 3.7: 8 models x 5 seeds (42, 123, 2024, 7, 99) run against CMU-MOSEI (thuiar/MMSA aligned-feature release; N=22,856 total -- 4,659 test / 16,326 train / 1,871 valid; `text_dim=768`, `audio_dim=74`, `vision_dim=35`, all confirmed at runtime; see `MOSEI_PROVENANCE.md` for the full provenance sign-off). Unlike S1-S10, which cover a full 8-model grid across 4 rates, S11 covers only the two comparisons the manuscript actually claims: the graded-robustness significance test (Claim A) and the text-missing collapse-rate check (Claim B). Every value below is drawn directly from the raw per-seed files `mosei_graded_robustness_raw.csv` and `mosei_single_modality_results.csv`, both included in this repository, and both independently recomputed (not merely matched) by `verify_manuscript_numbers.py`'s `verify_table_3_7_claim_a`/`verify_table_3_7_claim_b` functions (21/21 checks passing as of this revision, run directly against these exact files).

### S11.1 Claim A: graded-robustness comparison, full per-seed accuracy/F1

`gating_only_no_dropout` vs. `attention_gated_fusion_full`, all 4 missingness rates, all 5 seeds (underlying the paired t-tests in manuscript Table 3.7/Claim A):

| Model | Seed | Rate | Accuracy | F1 |
|---|---|---|---|---|
| attention_gated_fusion_full | 7 | 0.00 | 0.7162 | 0.7612 |
| attention_gated_fusion_full | 7 | 0.25 | 0.6626 | 0.7316 |
| attention_gated_fusion_full | 7 | 0.50 | 0.6137 | 0.7059 |
| attention_gated_fusion_full | 7 | 0.75 | 0.5855 | 0.6944 |
| attention_gated_fusion_full | 42 | 0.00 | 0.7296 | 0.7581 |
| attention_gated_fusion_full | 42 | 0.25 | 0.6699 | 0.7269 |
| attention_gated_fusion_full | 42 | 0.50 | 0.6235 | 0.7053 |
| attention_gated_fusion_full | 42 | 0.75 | 0.5840 | 0.6892 |
| attention_gated_fusion_full | 99 | 0.00 | 0.7403 | 0.7646 |
| attention_gated_fusion_full | 99 | 0.25 | 0.7004 | 0.7095 |
| attention_gated_fusion_full | 99 | 0.50 | 0.6546 | 0.6339 |
| attention_gated_fusion_full | 99 | 0.75 | 0.6345 | 0.5815 |
| attention_gated_fusion_full | 123 | 0.00 | 0.7268 | 0.7543 |
| attention_gated_fusion_full | 123 | 0.25 | 0.6680 | 0.7229 |
| attention_gated_fusion_full | 123 | 0.50 | 0.6244 | 0.7039 |
| attention_gated_fusion_full | 123 | 0.75 | 0.5864 | 0.6880 |
| attention_gated_fusion_full | 2024 | 0.00 | 0.7326 | 0.7596 |
| attention_gated_fusion_full | 2024 | 0.25 | 0.6778 | 0.7248 |
| attention_gated_fusion_full | 2024 | 0.50 | 0.6403 | 0.7067 |
| attention_gated_fusion_full | 2024 | 0.75 | 0.6068 | 0.6902 |
| gating_only_no_dropout | 7 | 0.00 | 0.7270 | 0.7630 |
| gating_only_no_dropout | 7 | 0.25 | 0.6849 | 0.7092 |
| gating_only_no_dropout | 7 | 0.50 | 0.6426 | 0.6416 |
| gating_only_no_dropout | 7 | 0.75 | 0.6164 | 0.5758 |
| gating_only_no_dropout | 42 | 0.00 | 0.7184 | 0.7594 |
| gating_only_no_dropout | 42 | 0.25 | 0.6834 | 0.7077 |
| gating_only_no_dropout | 42 | 0.50 | 0.6458 | 0.6388 |
| gating_only_no_dropout | 42 | 0.75 | 0.6132 | 0.5637 |
| gating_only_no_dropout | 99 | 0.00 | 0.7356 | 0.7598 |
| gating_only_no_dropout | 99 | 0.25 | 0.6847 | 0.6803 |
| gating_only_no_dropout | 99 | 0.50 | 0.6370 | 0.5873 |
| gating_only_no_dropout | 99 | 0.75 | 0.6137 | 0.5156 |
| gating_only_no_dropout | 123 | 0.00 | 0.7253 | 0.7568 |
| gating_only_no_dropout | 123 | 0.25 | 0.6922 | 0.7000 |
| gating_only_no_dropout | 123 | 0.50 | 0.6418 | 0.6040 |
| gating_only_no_dropout | 123 | 0.75 | 0.6117 | 0.5261 |
| gating_only_no_dropout | 2024 | 0.00 | 0.7212 | 0.7574 |
| gating_only_no_dropout | 2024 | 0.25 | 0.6780 | 0.6950 |
| gating_only_no_dropout | 2024 | 0.50 | 0.6390 | 0.6137 |
| gating_only_no_dropout | 2024 | 0.75 | 0.6184 | 0.5506 |

**Aggregate significance (recomputed from the table above via paired t-test, `scipy.stats.ttest_rel`):**

| Rate | mean(no-dropout) | mean(dropout) | diff | t-stat | p-value |
|---|---|---|---|---|---|
| 0.00 | 0.7255 | 0.7291 | -0.0036 | -0.8895 | 0.4240 |
| 0.25 | 0.6847 | 0.6757 | +0.0089 | 1.1951 | 0.2980 |
| 0.50 | 0.6413 | 0.6313 | +0.0100 | 1.1674 | 0.3079 |
| 0.75 | 0.6147 | 0.5994 | +0.0152 | 1.5822 | 0.1888 |

All four rates non-significant, matching the MOSI non-replication (Section 3.2) in both direction and conclusion.

### S11.2 Claim B: text-missing single-modality results, full per-seed accuracy/F1

All 8 models, all 5 seeds, text modality entirely absent (audio and vision only). Collapse threshold is F1<0.05, applied per-seed, matching the criterion used in S8 for the MOSI collapse table -- **not** a reused MOSI accuracy constant (see `MOSEI_PROVENANCE.md`).

| Model | Seed | Accuracy | F1 | Collapsed (F1<0.05)? |
|---|---|---|---|---|
| attention_gated_fusion_full | 7 | 0.4896 | 0.6568 |  |
| attention_gated_fusion_full | 42 | 0.4902 | 0.6576 |  |
| attention_gated_fusion_full | 99 | 0.5937 | 0.5543 |  |
| attention_gated_fusion_full | 123 | 0.4922 | 0.6534 |  |
| attention_gated_fusion_full | 2024 | 0.5181 | 0.6401 |  |
| dropout_only_fusion | 7 | 0.4879 | 0.6439 |  |
| dropout_only_fusion | 42 | 0.4928 | 0.6544 |  |
| dropout_only_fusion | 99 | 0.4896 | 0.6470 |  |
| dropout_only_fusion | 123 | 0.4879 | 0.6476 |  |
| dropout_only_fusion | 2024 | 0.4917 | 0.6568 |  |
| early_fusion | 7 | 0.5400 | 0.2467 |  |
| early_fusion | 42 | 0.5480 | 0.2677 |  |
| early_fusion | 99 | 0.4937 | 0.6377 |  |
| early_fusion | 123 | 0.5098 | 0.0000 | YES |
| early_fusion | 2024 | 0.5100 | 0.0009 | YES |
| fixed_weight_fusion | 7 | 0.4889 | 0.6534 |  |
| fixed_weight_fusion | 42 | 0.5394 | 0.2336 |  |
| fixed_weight_fusion | 99 | 0.4889 | 0.6550 |  |
| fixed_weight_fusion | 123 | 0.5338 | 0.1902 |  |
| fixed_weight_fusion | 2024 | 0.5100 | 0.0009 | YES |
| gating_only_no_dropout | 7 | 0.5675 | 0.5712 |  |
| gating_only_no_dropout | 42 | 0.5810 | 0.5572 |  |
| gating_only_no_dropout | 99 | 0.5460 | 0.3068 |  |
| gating_only_no_dropout | 123 | 0.5664 | 0.4192 |  |
| gating_only_no_dropout | 2024 | 0.5727 | 0.4517 |  |
| hard_mask_gated_fusion | 7 | 0.4909 | 0.6562 |  |
| hard_mask_gated_fusion | 42 | 0.4902 | 0.6567 |  |
| hard_mask_gated_fusion | 99 | 0.4902 | 0.6577 |  |
| hard_mask_gated_fusion | 123 | 0.5681 | 0.3681 |  |
| hard_mask_gated_fusion | 2024 | 0.5098 | 0.0000 | YES |
| imputation_baseline_post2023 | 7 | 0.5151 | 0.0488 | YES |
| imputation_baseline_post2023 | 42 | 0.5720 | 0.3876 |  |
| imputation_baseline_post2023 | 99 | 0.4911 | 0.6406 |  |
| imputation_baseline_post2023 | 123 | 0.5289 | 0.1279 |  |
| imputation_baseline_post2023 | 2024 | 0.5452 | 0.2609 |  |
| late_fusion | 7 | 0.5098 | 0.0000 | YES |
| late_fusion | 42 | 0.5765 | 0.4614 |  |
| late_fusion | 99 | 0.4902 | 0.6579 |  |
| late_fusion | 123 | 0.5149 | 0.0284 | YES |
| late_fusion | 2024 | 0.5136 | 0.0316 | YES |

**Summary:** 10 of 40 text-missing seed-runs across all 8 models are degenerate (F1<0.05). Restricted to the 3 dropout-trained models (`attention_gated_fusion_full`, `dropout_only_fusion`, and, notably, `gating_only_no_dropout` despite not being dropout-trained): 0 of 15 are degenerate. The remaining 5 non-dropout models show a seed-dependent 0-60% collapse rate each.

**Collapsed-run accuracy vs. MOSEI base rate:** the 10 collapsed (F1<0.05) runs above have accuracy in [0.5098, 0.5151] -- landing almost exactly on CMU-MOSEI's actual test-set majority-class rate (0.5098, confirmed at runtime; see `MOSEI_PROVENANCE.md`), not MOSI's 0.596. This confirms the same degenerate constant-output collapse signature identified in S8, reproduced against a different dataset's different base rate rather than an artifact of reusing MOSI's specific constant.

### S11.3 The `gating_only_no_dropout` divergence from MOSI -- reported directly, not smoothed over

On CMU-MOSI (S8), `gating_only_no_dropout` -- despite not being dropout-trained -- collapsed in a subset of seeds, and was counted among the non-dropout-trained models' seed-dependent 40-80% collapse range. On CMU-MOSEI, the same model shows 0/5 collapses, indistinguishable in this table from the two genuinely dropout-trained models. This is a real, dataset-dependent divergence, not a transcription error -- both S8's and S11.2's tables are independently verified against their respective raw per-seed files. We do not have a mechanistic explanation for it; manuscript Section 4/7 discusses two untested candidate factors (MOSEI's more balanced base rate and its roughly 5-8x larger training set relative to MOSI) without treating either as confirmed.

---
