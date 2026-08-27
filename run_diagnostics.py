"""
run_diagnostics.py — driver for Future Work item #1 (manuscript Section 7).

Loads checkpoints saved by run_experiment_grid.py (requires that script to
have been re-run with checkpoint saving enabled — see its docstring) and
produces:

  - single_modality_results.csv   real version of the old (removed) Table 3.4,
                                    for every model in config.yaml, all 3 seeds
  - gate_weights_raw.csv           per-sample gate weights vs. mask, for the
                                    two gated models, all 3 seeds, rates {0.25,0.5,0.75}
  - gate_weights_summary.csv       aggregated version of the above (real version
                                    of the old, removed Table 3.5)

Run this AFTER run_experiment_grid.py has been re-run with checkpoint saving.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.dataset import CmuMosiAligned
from diagnostics import (
    load_checkpoint,
    single_modality_masking_eval,
    gate_weight_analysis,
    summarize_gate_weights,
)
from train import GATED_MODELS


def run_diagnostics(config_path: str = "config.yaml", checkpoint_dir: str = "checkpoints"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(checkpoint_dir)

    ckpt_paths = sorted(checkpoint_dir.glob("*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}/. Re-run run_experiment_grid.py "
            "(with the checkpoint-saving version) before running diagnostics — "
            "see run_experiment_grid.py's docstring."
        )
    print(f"Found {len(ckpt_paths)} checkpoints in {checkpoint_dir}/")

    test_ds = CmuMosiAligned(data_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    single_modality_rows = []
    gate_weight_rows = []

    for ckpt_path in ckpt_paths:
        encoders, fusion_model, model_name, seed, is_gated = load_checkpoint(ckpt_path, device)
        print(f"=== diagnostics: {model_name} seed={seed} ===")

        # 1. Single-modality masking eval — run for every model (cheap, and lets
        #    us compare modality-importance patterns across the whole grid, not
        #    just the two gated models).
        smr = single_modality_masking_eval(
            encoders, fusion_model, test_loader, device, is_gated, model_name, seed
        )
        single_modality_rows.append(smr)
        print(smr.to_string(index=False))

        # 2. Gate weight analysis — only meaningful for gated models.
        if is_gated:
            gwr = gate_weight_analysis(encoders, fusion_model, test_loader, device, model_name, seed)
            gate_weight_rows.append(gwr)

    single_modality_df = pd.concat(single_modality_rows, ignore_index=True)
    single_modality_df.to_csv("single_modality_results.csv", index=False)
    print(f"\nWrote single_modality_results.csv ({len(single_modality_df)} rows)")

    if gate_weight_rows:
        gate_weights_raw = pd.concat(gate_weight_rows, ignore_index=True)
        gate_weights_raw.to_csv("gate_weights_raw.csv", index=False)
        print(f"Wrote gate_weights_raw.csv ({len(gate_weights_raw)} rows)")

        gate_weights_summary = summarize_gate_weights(gate_weights_raw)
        gate_weights_summary.to_csv("gate_weights_summary.csv", index=False)
        print(f"Wrote gate_weights_summary.csv ({len(gate_weights_summary)} rows)")
        print(gate_weights_summary.to_string(index=False))
    else:
        print("No gated-model checkpoints found — skipping gate weight analysis.")

    return single_modality_df, (pd.concat(gate_weight_rows, ignore_index=True) if gate_weight_rows else None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()
    run_diagnostics(args.config, args.checkpoint_dir)
