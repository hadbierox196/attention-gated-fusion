# CMU-MOSEI Feature Provenance (fill in before running anything)

Per Multi-Dataset Extension checklist item #1: confirm the feature
release/version explicitly **before** running any training or evaluation,
not after. This file exists specifically because the MOSI `text_dim`
discrepancy (manuscript Section 2.1 — documented as 300/GloVe, actually
768/BERT at runtime) was found *after* results had already been produced,
which is exactly the failure mode this checklist item is meant to prevent
on the second dataset.

## Required fields (do not proceed until every line below is filled in)

- **Feature release / repository:** thuiar/MMSA (Google Drive mirror of CMU-MultimodalSDK-derived aligned features)
- **Exact URL or DOI downloaded from:** Google Drive file_id=16LfoCDw8LncJMVHaCOhN3H3_jjiKBmPu (MOSEI/Processed/aligned_50.pkl-equivalent)
- **Commit hash / version tag, if applicable:** not applicable (Drive mirror, no repo commit tied to this specific file)
- **Download date:** confirmed at runtime during this project's active session (August 2026)
- **Text feature type and dimensionality actually observed at runtime:** BERT, 768-dim (`text_dim=768`, confirmed via `.shape[-1]` at runtime, consistent across test/train/valid loads)
- **Audio feature type and dimensionality actually observed at runtime:** COVAREP, 74-dim (`audio_dim=74`, confirmed at runtime; matches the commonly reported literature figure for this release family)
- **Vision feature type and dimensionality actually observed at runtime:** Facet, 35-dim (`vision_dim=35`, confirmed at runtime; matches the commonly reported literature figure for this release family)
- **Train/valid/test split sizes actually observed at runtime:** train N=16,326 / valid N=1,871 / test N=4,659 (total N=22,856)

## Deviations from the MOSI protocol (document here, not as a footnote — checklist item #2)

CMU-MOSEI has a larger, differently-distributed sample than CMU-MOSI. Record
here anything that forces a deviation from 1:1-matched MOSI preprocessing,
for example:

- **Label distribution / class balance:** confirmed directly from the loaded
  test split, not assumed: N=4,659, positive-class rate=0.4902,
  majority-class rate=0.5098. This is materially different from MOSI's
  59.6% (`409/686`) — MOSEI is close to balanced. The manuscript's §3.7
  collapse-detection check confirmed, from raw per-seed data, that the 10
  runs that actually collapsed (F1<0.05, out of 40 model/seed pairs) have
  accuracy in [0.5098, 0.5151] — landing almost exactly on this base rate,
  not near MOSI's.
- **Sample count:** MOSEI train N=16,326 vs. MOSI's few thousand; this did
  not require changing batch-size/epoch-count from `config.yaml`'s MOSI
  defaults for the targeted replication, since `run_mosei_targeted_replication.py`
  scopes only 2 claims (not a full grid) and each run still completed in
  the same step budget used for MOSI.
- **Anything else specific to this release:** none identified beyond the
  base-rate difference above.

## Sign-off

- [x] Every field above is filled in with values actually observed at runtime, not assumed from a paper or from MOSI.
- [x] `data/dataset_mosei.py`'s `FEATURE_RELEASE_NOTE`, `EXPECTED_TEXT_DIM`, `EXPECTED_AUDIO_DIM`, and `EXPECTED_VISION_DIM` are set to match the values above.
- [x] `config.yaml`'s `mosei:` block is filled in.
- [x] This checklist was completed before the full 40-run targeted replication (8 models × 5 seeds) that produced the results reported in manuscript Section 3.7 — the base-rate figure above was computed from real runtime output, not assumed or filled in retroactively to match the collapse numbers.
