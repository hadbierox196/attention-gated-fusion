"""
Full experiment grid driver — Section 2.7.

Runs all 7 models x 3 seeds = 21 training runs, then evaluates each trained
model at all 4 missingness rates (0.00, 0.25, 0.50, 0.75), producing the
7 x 3 x 4 = 84 evaluation points reported in Table 3.1.

Writes:
  - results_raw.csv     (one row per model x seed x missingness_rate)
  - config_log.json     (full config + environment info for each run, for
                          reproducibility per Section 2.7)
  - checkpoints/*.pt     (best-val-F1 checkpoint per (model, seed), so
                          diagnostics.py / run_diagnostics.py can load
                          trained models directly instead of retraining —
                          needed for the Future Work #1 mechanism analysis)

CONFIRMED: this has been executed against real CMU-MOSI data; see manuscript
Section 3 and the archived results at https://doi.org/10.5281/zenodo.22105162.
Checkpoint saving was added after that run, so re-running this script (e.g.
to include hard_mask_gated_fusion, per config.yaml) is required before
run_diagnostics.py has checkpoints to load from.
"""

from __future__ import annotations
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import CmuMosiAligned, binarize
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS, train_one_run, set_seed


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


@torch.no_grad()
def evaluate_at_rate(encoders, fusion_model, loader, device, is_gated, rate, seed_for_masking):
    encoders.eval()
    fusion_model.eval()
    gen = torch.Generator(device=device).manual_seed(seed_for_masking)

    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_batch, mask = apply_missingness(batch, rate, generator=gen)
        embeddings = encoders(masked_batch)
        if is_gated:
            pred, _ = fusion_model(embeddings, mask)
        else:
            pred = fusion_model(embeddings, mask)
        all_preds.append((pred > 0).cpu())
        all_labels.append(binarize(batch["label"]).cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "n": len(labels),
    }


def run_grid(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["path"]
    if not data_path:
        raise ValueError(
            "config.yaml: data.path is not set. Point it at your local aligned_50.pkl "
            "before running the grid (see README / manuscript Section 9)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = config["hidden_dim"]

    results_rows = []
    config_log = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": device,
        "git_commit": _git_commit_hash(),
        "config": config,
        "runs": [],
    }

    test_ds = CmuMosiAligned(data_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    checkpoint_dir = Path(config.get("output", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for model_name in config["models"]:
        for seed in config["seeds"]:
            print(f"=== training {model_name} seed={seed} ===")
            best_state, best_val_f1 = train_one_run(
                model_name=model_name,
                seed=seed,
                data_path=data_path,
                hidden_dim=hidden_dim,
                lr=config["learning_rate"],
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                device=device,
            )

            # Rebuild models and load best checkpoint for evaluation
            set_seed(seed)
            encoders = TriModalEncoders(
                text_dim=test_ds.text.shape[-1],
                audio_dim=test_ds.audio_dim,
                vision_dim=test_ds.vision_dim,
                hidden_dim=hidden_dim,
            ).to(device)
            fusion_model = MODEL_REGISTRY[model_name](hidden_dim=hidden_dim).to(device)
            encoders.load_state_dict(best_state["encoders"])
            fusion_model.load_state_dict(best_state["fusion_model"])

            # Save checkpoint to disk so diagnostics.py can load this exact
            # trained model later without retraining (Future Work #1).
            ckpt_path = checkpoint_dir / f"{model_name}__seed{seed}.pt"
            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "encoders_state_dict": best_state["encoders"],
                    "fusion_model_state_dict": best_state["fusion_model"],
                    "best_val_f1": best_val_f1,
                    "text_dim": test_ds.text.shape[-1],
                    "audio_dim": test_ds.audio_dim,
                    "vision_dim": test_ds.vision_dim,
                    "hidden_dim": hidden_dim,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint -> {ckpt_path}")

            is_gated = model_name in GATED_MODELS

            for rate in config["missingness"]["eval_rates"]:
                # Fixed masking seed per (model, seed, rate) so all models see the
                # same masking pattern at a given rate for a given seed — Section 2.4.
                masking_seed = seed * 1000 + int(rate * 100)
                metrics = evaluate_at_rate(
                    encoders, fusion_model, test_loader, device, is_gated, rate, masking_seed
                )
                results_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "missingness_rate": rate,
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "n_test": metrics["n"],
                        "best_val_f1": best_val_f1,
                    }
                )
                print(f"  rate={rate}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

            config_log["runs"].append(
                {
                    "model": model_name,
                    "seed": seed,
                    "best_val_f1": best_val_f1,
                    "n_params": sum(p.numel() for p in encoders.parameters())
                    + sum(p.numel() for p in fusion_model.parameters()),
                }
            )

    results_df = pd.DataFrame(results_rows)
    out_csv = config["output"]["results_csv"]
    out_json = config["output"]["config_log_json"]
    results_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(config_log, f, indent=2)

    print(f"\nWrote {out_csv} ({len(results_df)} rows) and {out_json}")
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_grid(args.config)
