"""
Mechanism-level diagnostics — Future Work item #1 (manuscript Section 7).

This is the real version of what the manuscript's earlier (fabricated, since
removed) Sections 3.4-3.6 claimed to show without ever having been run. Two
analyses:

1. single_modality_masking_eval: mask exactly ONE modality at a time (not a
   uniform random rate) and evaluate accuracy — answers "how costly is losing
   text specifically, vs. audio, vs. vision, for a given trained model?"

2. gate_weight_analysis: for gated models only, record the gate's per-sample
   softmax weights alongside the missingness mask, at several eval rates —
   answers "does the gate actually reweight away from a masked-out modality,
   and does that behavior differ between the dropout-trained and
   non-dropout-trained gate?" This is the direct test that can distinguish
   Hypothesis A (training-signal dilution) from Hypothesis B (encoder
   disruption) in manuscript Section 4: if the dropout-trained gate's WEIGHTS
   look similarly mask-responsive to the no-dropout gate's, the degradation
   is more likely downstream in the encoders (Hypothesis B); if the
   dropout-trained gate's weights look flatter / less mask-responsive, that
   supports Hypothesis A directly.

Both write raw per-sample CSVs (not just aggregates) so a reader can
recompute any summary statistic independently, consistent with the
manuscript's disclosed provenance standard (Section 5).
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch

from data.dataset import CmuMosiAligned, binarize
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS

MODALITIES = ("text", "audio", "vision")


def load_checkpoint(checkpoint_path: str | Path, device: str):
    """Rebuild encoders + fusion model from a checkpoint saved by run_experiment_grid.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]

    encoders = TriModalEncoders(
        text_dim=ckpt["text_dim"],
        audio_dim=ckpt["audio_dim"],
        vision_dim=ckpt["vision_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    encoders.load_state_dict(ckpt["encoders_state_dict"])

    fusion_model = MODEL_REGISTRY[model_name](hidden_dim=ckpt["hidden_dim"]).to(device)
    fusion_model.load_state_dict(ckpt["fusion_model_state_dict"])

    is_gated = model_name in GATED_MODELS
    return encoders, fusion_model, model_name, ckpt["seed"], is_gated


def _apply_single_modality_mask(batch: dict, missing_modality: str, device: str):
    """
    Build a mask that zeroes exactly ONE modality and keeps the other two fully
    present — unlike apply_missingness, which applies an independent per-modality
    Bernoulli draw. This directly answers "how much does losing modality X alone
    cost this model?" for a real version of the old Table 3.4.
    """
    B = batch["text"].shape[0]
    mask = torch.ones(B, len(MODALITIES), device=device)
    idx = MODALITIES.index(missing_modality)
    mask[:, idx] = 0.0

    masked_batch = {}
    for i, name in enumerate(MODALITIES):
        m = mask[:, i].view(B, 1, 1)
        masked_batch[name] = batch[name] * m
    return masked_batch, mask


@torch.no_grad()
def single_modality_masking_eval(
    encoders, fusion_model, loader, device, is_gated, model_name: str, seed: int
) -> pd.DataFrame:
    """Evaluate accuracy with each single modality masked out, one at a time, plus a full-input baseline."""
    from sklearn.metrics import accuracy_score, f1_score

    encoders.eval()
    fusion_model.eval()

    rows = []
    conditions = ["none"] + list(MODALITIES)  # "none" = no modality masked (baseline)

    for missing in conditions:
        all_preds, all_labels = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if missing == "none":
                B = batch["text"].shape[0]
                mask = torch.ones(B, len(MODALITIES), device=device)
                masked_batch = {k: batch[k] for k in MODALITIES}
            else:
                masked_batch, mask = _apply_single_modality_mask(batch, missing, device)

            embeddings = encoders(masked_batch)
            if is_gated:
                pred, _ = fusion_model(embeddings, mask)
            else:
                pred = fusion_model(embeddings, mask)
            all_preds.append((pred > 0).cpu())
            all_labels.append(binarize(batch["label"]).cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "missing_modality": missing,
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "n": len(labels),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def gate_weight_analysis(
    encoders, gate_fusion_model, loader, device, model_name: str, seed: int, rates=(0.25, 0.5, 0.75)
) -> pd.DataFrame:
    """
    For GATED models only. At each rate, apply the standard apply_missingness
    protocol (same as Table 3.1), record the gate's softmax weight for each
    modality on every sample, alongside whether that modality was masked for
    that sample. Returns one row per (sample, rate) — fully raw, not pre-aggregated.
    """
    encoders.eval()
    gate_fusion_model.eval()

    rows = []
    for rate in rates:
        gen = torch.Generator(device=device).manual_seed(seed * 1000 + int(rate * 100))
        sample_idx = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            masked_batch, mask = apply_missingness(batch, rate, generator=gen)
            embeddings = encoders(masked_batch)
            _, weights = gate_fusion_model(embeddings, mask)  # weights: [B, 3] in (text, audio, vision) order

            weights_cpu = weights.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            B = weights_cpu.shape[0]
            for i in range(B):
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "rate": rate,
                        "sample_idx": sample_idx,
                        "w_text": weights_cpu[i, 0],
                        "w_audio": weights_cpu[i, 1],
                        "w_vision": weights_cpu[i, 2],
                        "text_present": int(mask_cpu[i, 0]),
                        "audio_present": int(mask_cpu[i, 1]),
                        "vision_present": int(mask_cpu[i, 2]),
                    }
                )
                sample_idx += 1
    return pd.DataFrame(rows)


def summarize_gate_weights(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean gate weight per modality, split by whether that modality was present,
    per (model, rate) — the real version of the old (fabricated) Section 3.5 table.
    """
    rows = []
    for (model, rate), group in raw_df.groupby(["model", "rate"]):
        for mod, w_col, present_col in [
            ("text", "w_text", "text_present"),
            ("audio", "w_audio", "audio_present"),
            ("vision", "w_vision", "vision_present"),
        ]:
            for present_val, label in [(1, "present"), (0, "absent")]:
                subset = group[group[present_col] == present_val]
                if len(subset) == 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "rate": rate,
                        "modality": mod,
                        "status": label,
                        "mean_weight": subset[w_col].mean(),
                        "std_weight": subset[w_col].std(),
                        "n": len(subset),
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "rate", "modality", "status"])
