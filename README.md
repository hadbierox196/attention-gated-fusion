# Attention-Gated Fusion for Modality-Robust Affective Computing — Code Scaffold

**Status: scaffold, verified against real CMU-MOSI data once.** The core pipeline (`run_experiment_grid.py`) has been run end-to-end and produced the results reported in the manuscript's Section 3 — see `results_raw.csv` and `config_log.json`, archived at https://doi.org/10.5281/zenodo.22105162. Checkpoint saving and the mechanism-diagnostics modules (`diagnostics.py`, `run_diagnostics.py`) were added after that run and have **not yet been executed** — running them is the next step (Future Work #1/#2, manuscript Section 7).

## Structure

```
code_scaffold/
├── README.md                  # this file
├── requirements.txt
├── config.yaml                # training config matching Section 2.5
├── data/
│   └── dataset.py              # CMU-MOSI aligned_50.pkl loading + preprocessing (Section 2.2)
├── models/
│   ├── encoders.py             # shared GRU modality encoders (Section 2.3)
│   ├── fusion.py                # all 7 fusion variants (Section 2.3 baseline table)
│   ├── gate.py                  # attention-gated fusion module (Section 2.3)
│   └── hard_mask_gate.py        # hard-mask gate variant (Future Work #2)
├── missingness.py              # apply_missingness(batch, rate) (Section 2.4)
├── train.py                     # single (model, seed) training run
├── run_experiment_grid.py       # drives the full 8×3×4 grid, writes results_raw.csv, config_log.json, checkpoints/
├── diagnostics.py               # single-modality masking eval + gate-weight logging (Future Work #1)
├── run_diagnostics.py           # driver: loads checkpoints, runs diagnostics.py, writes CSVs
└── evaluate.py                   # significance tests (Section 3.2), efficiency table (Section 3.3)
```

## Run order

1. `run_experiment_grid.py` — trains all 8 models × 3 seeds, writes `results_raw.csv`, `config_log.json`, and `checkpoints/*.pt`.
2. `evaluate.py` — significance tests / CIs / efficiency table from `results_raw.csv`.
3. `run_diagnostics.py` — **requires step 1's checkpoints.** Writes `single_modality_results.csv`, `gate_weights_raw.csv`, `gate_weights_summary.csv` — the real version of the manuscript's originally-fabricated, since-removed Sections 3.4–3.6.

## What this scaffold deliberately does NOT include

- The real CMU-MOSI data. `data/dataset.py` expects `aligned_50.pkl` at a path you provide; it
  does not download or redistribute it (see manuscript Section 9, Data Availability).
- Any pre-computed results. Running `run_experiment_grid.py` will produce a fresh
  `results_raw.csv` — it will only match the manuscript's Table 3.1 if the implementation here
  is faithful to what actually produced those numbers, which the author must verify (step 1 above).
- A working `imputation_baseline_post2023` reconstruction sub-network beyond a minimal stub —
  the manuscript is explicit (Section 2.3) that this is a representative reimplementation of the
  DAST-GAN family's reconstruction-before-fusion principle, not a reproduction of DAST-GAN itself,
  and the stub here should be reviewed against whatever reconstruction architecture was actually
  used for the reported numbers.
