"""
Re-trains the dropout-relevant models under a narrower training-missingness
range (Uniform(0, 0.4) instead of Uniform(0, 0.75)) to test whether the
"flatter baseline policy" effect (manuscript Section 4, Hypothesis A refined)
is sensitive to the training range's width. Manuscript Section 7, item 4.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from data.dataset import CmuMosiAligned, binarize
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS, DROPOUT_TRAINED, set_seed

NARROW_RANGE = (0.0, 0.4)


def train_one_run_narrow(model_name, seed, data_path, hidden_dim=128, lr=1e-3,
                          batch_size=32, epochs=15,
                          device="cuda" if torch.cuda.is_available() else "cpu"):
    set_seed(seed)
    train_ds = CmuMosiAligned(data_path, split="train")
    val_ds = CmuMosiAligned(data_path, split="valid")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    encoders = TriModalEncoders(
        text_dim=train_ds.text.shape[-1], audio_dim=train_ds.audio_dim,
        vision_dim=train_ds.vision_dim, hidden_dim=hidden_dim,
    ).to(device)
    fusion_model = MODEL_REGISTRY[model_name](hidden_dim=hidden_dim).to(device)
    params = list(encoders.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()
    use_train_dropout = model_name in DROPOUT_TRAINED
    is_gated = model_name in GATED_MODELS

    best_val_f1, best_state = -1.0, None
    for epoch in range(epochs):
        encoders.train(); fusion_model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            rate = float(torch.empty(1).uniform_(*NARROW_RANGE).item()) if use_train_dropout else 0.0
            masked_batch, mask = apply_missingness(batch, rate)
            embeddings = encoders(masked_batch)
            out = fusion_model(embeddings, mask)
            pred = out[0] if is_gated else out
            loss = loss_fn(pred, batch["label"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        encoders.eval(); fusion_model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                masked_batch, mask = apply_missingness(batch, 0.0)
                embeddings = encoders(masked_batch)
                out = fusion_model(embeddings, mask)
                pred = out[0] if is_gated else out
                preds.append((pred > 0).cpu()); labels.append(binarize(batch["label"]).cpu())
            val_f1 = f1_score(torch.cat(labels).numpy(), torch.cat(preds).numpy())
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {"encoders": {k: v.cpu().clone() for k, v in encoders.state_dict().items()},
                           "fusion_model": {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}}
    return best_state, best_val_f1


@torch.no_grad()
def _evaluate_at_rate(encoders, fusion_model, loader, device, is_gated, rate, seed_for_masking):
    encoders.eval(); fusion_model.eval()
    gen = torch.Generator(device=device).manual_seed(seed_for_masking)
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_batch, mask = apply_missingness(batch, rate, generator=gen)
        embeddings = encoders(masked_batch)
        out = fusion_model(embeddings, mask)
        pred = out[0] if is_gated else out
        preds.append((pred > 0).cpu()); labels.append(binarize(batch["label"]).cpu())
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"accuracy": accuracy_score(l, p), "f1": f1_score(l, p), "n": len(l)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--models", nargs="+", default=["dropout_only_fusion", "attention_gated_fusion_full", "hard_mask_gated_fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 99])
    parser.add_argument("--eval_rates", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--out_csv", default="results_narrow_dropout_range.csv")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds = CmuMosiAligned(args.data_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"=== [narrow range 0-0.4] training {model_name} seed={seed} ===")
            best_state, best_val_f1 = train_one_run_narrow(model_name, seed, args.data_path)
            set_seed(seed)
            encoders = TriModalEncoders(text_dim=test_ds.text.shape[-1], audio_dim=test_ds.audio_dim,
                                         vision_dim=test_ds.vision_dim, hidden_dim=128).to(device)
            fusion_model = MODEL_REGISTRY[model_name](hidden_dim=128).to(device)
            encoders.load_state_dict(best_state["encoders"])
            fusion_model.load_state_dict(best_state["fusion_model"])
            is_gated = model_name in GATED_MODELS
            for rate in args.eval_rates:
                masking_seed = seed * 1000 + int(rate * 100)
                m = _evaluate_at_rate(encoders, fusion_model, test_loader, device, is_gated, rate, masking_seed)
                rows.append({"model": model_name, "seed": seed, "missingness_rate": rate,
                             "train_rate_range": "0.0-0.4", **m})
                print(f"  rate={rate}: acc={m['accuracy']:.4f}")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
