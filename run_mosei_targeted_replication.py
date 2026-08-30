"""
run_mosei_targeted_replication.py — Multi-Dataset Extension checklist items
3, 4, 5.

This deliberately does NOT reproduce the full 8-model x 5-seed x 4-rate MOSI
grid on CMU-MOSEI. Per checklist item #4/#5, it tests only the two claims
this manuscript's Abstract-adjacent "current claims at a glance" box treats
as live:

  (A) The graded-robustness non-effect (Section 3.2): does modality-dropout
      training show no significant benefit to an attention gate's graded
      missingness robustness on MOSEI either? Compares
      `gating_only_no_dropout` vs `attention_gated_fusion_full` at each of
      the four missingness rates, paired t-test across 5 seeds -- the same
      comparison and same n as the MOSI Table 3.2b analysis.

  (B) The collapse-prevention effect (Section 3.4): does the same
      dropout-trained / non-dropout-trained collapse split appear under
      complete loss of the dominant modality (text) on MOSEI? Evaluates all
      8 models, all 5 seeds, text-missing condition only, accuracy AND F1
      (checklist item #7 -- checked from this script's first run, not added
      after the fact).

Uses 5 seeds directly (checklist item #3) -- NOT a 3-seed pilot -- given
this manuscript's own Section 1.1/4 account of what happened when the
original MOSI ablation was trusted at n=3.

Before running: fill in data/dataset_mosei.py's FEATURE_RELEASE_NOTE and
EXPECTED_*_DIM constants (checklist item #1/#8), and config.yaml's `mosei:`
block. This script will refuse to run otherwise.

Writes:
  - mosei_graded_robustness_significance.csv   (claim A, Table-3.2b-equivalent)
  - mosei_single_modality_results.csv          (claim B, Table-3.4-equivalent, incl. F1)

Does NOT produce a Table-3.1-equivalent (all 8 models x 4 rates x 5 seeds
accuracy grid) -- that is explicitly out of scope per checklist item #4's
"don't try to replicate every result" guidance. If a reviewer specifically
asks for it, run run_experiment_grid.py with config.yaml's mosei block
instead; this script exists to keep the targeted claims honest and fast to
check, not to be the only cross-dataset artifact possible.

STATUS AS OF THIS REVISION: this script has been written and is ready to
run, but has NOT been executed against real CMU-MOSEI data -- no such data
file is available in this environment. Manuscript Section 4.X and the
"current claims at a glance" box both mark cross-dataset replication as
PENDING, not completed, until someone runs this and reports the output.
Do not fill in placeholder numbers to make those sections look finished.
"""

from __future__ import annotations
import argparse

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS, train_one_run


@torch.no_grad()
def _evaluate(encoders, fusion_model, loader, device, is_gated, rate, binarize_fn, seed_for_masking=None):
    encoders.eval()
    fusion_model.eval()
    gen = None
    if seed_for_masking is not None:
        gen = torch.Generator(device=device).manual_seed(seed_for_masking)
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_batch, mask = apply_missingness(batch, rate, generator=gen)
        embeddings = encoders(masked_batch)
        pred = fusion_model(embeddings, mask)
        if is_gated:
            pred = pred[0]
        all_preds.append((pred > 0).cpu())
        all_labels.append(binarize_fn(batch["label"]).cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return accuracy_score(labels, preds), f1_score(labels, preds), len(labels)


def run_targeted_replication(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mosei_cfg = config.get("mosei", {})
    data_path = mosei_cfg.get("path")
    if not data_path:
        raise ValueError(
            "config.yaml: mosei.path is not set. Point it at your local MOSEI "
            "feature pickle first (checklist item #1)."
        )

    # Import here, not at module load time, so the FEATURE_RELEASE_NOTE
    # guard in dataset_mosei.py is the first thing that can fail loudly,
    # rather than an unrelated import error masking it.
    from data.dataset_mosei import CmuMoseiAligned
    from data.dataset import binarize  # same convention, reused per item #2

    seeds = mosei_cfg.get("seeds", [42, 123, 2024, 7, 99])
    if len(seeds) < 5:
        raise ValueError(
            f"checklist item #3: use 5 seeds directly, not a 3-seed pilot. "
            f"config.yaml mosei.seeds currently has {len(seeds)}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = config["hidden_dim"]
    eval_rates = config["missingness"]["eval_rates"]  # same as MOSI, item #2

    single_modality_rows = []
    graded_rows = []

    test_ds = CmuMoseiAligned(data_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    for seed in seeds:
        for model_name in MODEL_REGISTRY:
            is_gated = model_name in GATED_MODELS
            best_state, best_val_f1 = train_one_run(
                model_name, seed, data_path, hidden_dim=hidden_dim,
                lr=config["learning_rate"], batch_size=config["batch_size"],
                epochs=config["epochs"], device=device, dataset_cls=CmuMoseiAligned,
            )

            encoders = TriModalEncoders(
                text_dim=test_ds.text.shape[-1], audio_dim=test_ds.audio_dim,
                vision_dim=test_ds.vision_dim, hidden_dim=hidden_dim,
            ).to(device)
            fusion_model = MODEL_REGISTRY[model_name](hidden_dim=hidden_dim).to(device)
            encoders.load_state_dict(best_state["encoders"])
            fusion_model.load_state_dict(best_state["fusion_model"])

            # Claim B: text-missing condition. accuracy AND F1 from the
            # first run (checklist item #7), matching Table 3.4's exact
            # criterion for a collapse (F1 < 0.05) used below.
            masked_batch = {
                "text": torch.zeros_like(test_ds.text), "audio": test_ds.audio,
                "vision": test_ds.vision, "label": test_ds.labels,
            }
            acc, f1, n = _evaluate(
                encoders, fusion_model, [masked_batch], device, is_gated,
                rate=0.0, binarize_fn=binarize,
            )
            single_modality_rows.append({
                "model": model_name, "seed": seed, "missing_modality": "text",
                "accuracy": acc, "f1": f1, "n": n,
            })

            # Claim A: computed for every model (cheap, already have the
            # checkpoint loaded) so evaluate.py's paired_ttest_by_rate can
            # be reused unmodified on the two gate models below.
            for rate in eval_rates:
                acc_r, f1_r, n_r = _evaluate(
                    encoders, fusion_model, test_loader, device, is_gated,
                    rate=rate, binarize_fn=binarize, seed_for_masking=seed,
                )
                graded_rows.append({
                    "model": model_name, "seed": seed, "missingness_rate": rate,
                    "accuracy": acc_r, "f1": f1_r, "n_test": n_r, "best_val_f1": best_val_f1,
                })

    single_df = pd.DataFrame(single_modality_rows)
    graded_df = pd.DataFrame(graded_rows)
    single_df.to_csv("mosei_single_modality_results.csv", index=False)
    graded_df.to_csv("mosei_graded_robustness_raw.csv", index=False)

    from evaluate import paired_ttest_by_rate
    sig_df = paired_ttest_by_rate(graded_df, "gating_only_no_dropout", "attention_gated_fusion_full")
    sig_df.to_csv("mosei_graded_robustness_significance.csv", index=False)

    single_df["collapsed"] = single_df["f1"] < 0.05
    collapse_summary = single_df.groupby("model")["collapsed"].agg(["sum", "count"])
    collapse_summary["rate"] = collapse_summary["sum"] / collapse_summary["count"]

    print("=== Claim A: graded-robustness significance (MOSEI) ===")
    print(sig_df.to_string(index=False))
    print("\n=== Claim B: text-missing collapse rate by model (MOSEI) ===")
    print(collapse_summary.to_string())
    print(
        "\nCompare directly against manuscript Table 3.2b (MOSI) and Section "
        "3.4's 0/15 dropout-trained vs. 40-80% non-dropout-trained collapse "
        "split. Report whichever way this comes out (checklist item #10) -- "
        "do not soften a MOSEI non-replication relative to how the original "
        "MOSI non-replication was reported in Section 3.2."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_targeted_replication(args.config)
