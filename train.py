"""
Single (model, seed) training run — Section 2.5.

This is the unit that run_experiment_grid.py calls 21 times (7 models x 3 seeds)
to reproduce the full grid described in Section 2.7.
"""

from __future__ import annotations
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import CmuMosiAligned, binarize
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from models.fusion import (
    EarlyFusion,
    LateFusion,
    FixedWeightFusion,
    DropoutOnlyFusion,
    ImputationBaselinePost2023,
)
from models.gate import AttentionGatedFusion
from models.hard_mask_gate import HardMaskGatedFusion


def set_seed(seed: int):
    """Fixes Python, NumPy, PyTorch, and CUDA seeds — Section 2.5."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODEL_REGISTRY = {
    "early_fusion": EarlyFusion,
    "late_fusion": LateFusion,
    "fixed_weight_fusion": FixedWeightFusion,
    "dropout_only_fusion": DropoutOnlyFusion,
    "gating_only_no_dropout": AttentionGatedFusion,  # same arch, dropout=False at train time
    "attention_gated_fusion_full": AttentionGatedFusion,
    "imputation_baseline_post2023": ImputationBaselinePost2023,
    "hard_mask_gated_fusion": HardMaskGatedFusion,  # Future Work item #2 (Section 7)
}

# Models trained WITH modality dropout at train time (Section 2.4/2.3 baseline table)
DROPOUT_TRAINED = {"dropout_only_fusion", "attention_gated_fusion_full", "hard_mask_gated_fusion"}

# Models whose forward() returns (pred, gate_weights) instead of just pred
GATED_MODELS = {"gating_only_no_dropout", "attention_gated_fusion_full", "hard_mask_gated_fusion"}


def train_one_run(
    model_name: str,
    seed: int,
    data_path: str,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(seed)

    train_ds = CmuMosiAligned(data_path, split="train")
    val_ds = CmuMosiAligned(data_path, split="valid")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    encoders = TriModalEncoders(
        text_dim=train_ds.text.shape[-1],
        audio_dim=train_ds.audio_dim,
        vision_dim=train_ds.vision_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    fusion_model = MODEL_REGISTRY[model_name](hidden_dim=hidden_dim).to(device)

    params = list(encoders.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    use_train_dropout = model_name in DROPOUT_TRAINED
    is_gated = model_name in GATED_MODELS

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        encoders.train()
        fusion_model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_train_dropout:
                # Train-time missingness rate resampled per batch from Uniform(0, 0.75) — Section 2.4
                rate = float(torch.empty(1).uniform_(0.0, 0.75).item())
            else:
                rate = 0.0
            masked_batch, mask = apply_missingness(batch, rate)

            embeddings = encoders(masked_batch)
            if is_gated:
                pred, _ = fusion_model(embeddings, mask)
            else:
                pred = fusion_model(embeddings, mask)

            loss = loss_fn(pred, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- validation (best-F1 checkpointing — Section 2.5) ---
        val_f1 = _evaluate_f1(encoders, fusion_model, val_loader, device, is_gated, rate=0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "encoders": {k: v.cpu().clone() for k, v in encoders.state_dict().items()},
                "fusion_model": {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()},
            }

    return best_state, best_val_f1


@torch.no_grad()
def _evaluate_f1(encoders, fusion_model, loader, device, is_gated, rate: float):
    from sklearn.metrics import f1_score

    encoders.eval()
    fusion_model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_batch, mask = apply_missingness(batch, rate)
        embeddings = encoders(masked_batch)
        if is_gated:
            pred, _ = fusion_model(embeddings, mask)
        else:
            pred = fusion_model(embeddings, mask)
        all_preds.append((pred > 0).cpu())
        all_labels.append(binarize(batch["label"]).cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return f1_score(labels, preds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    state, val_f1 = train_one_run(args.model, args.seed, args.data_path)
    print(f"model={args.model} seed={args.seed} best_val_f1={val_f1:.4f}")
