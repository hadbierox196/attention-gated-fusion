"""
apply_missingness(batch, rate) — Section 2.4 of the manuscript.

Used identically at train and test time to avoid train/test protocol mismatch.
Each modality is independently retained with probability (1 - rate); if a draw
would zero out all three modalities for a sample, one modality is force-kept
at random so at least one modality is always present. Masked-out modalities
have their entire feature tensor zeroed — shape is preserved, only values change.
"""

from __future__ import annotations
import torch


def apply_missingness(
    batch: dict[str, torch.Tensor],
    rate: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Args:
        batch: dict with keys 'text', 'audio', 'vision', each a
            [B, T, D_modality] float32 tensor.
        rate: float in [0, 1), probability each modality is masked out
            independently per sample.
        generator: optional torch.Generator for reproducible masking
            (seed identically per model/seed/rate combination — Section 2.4,
            "Evaluation-time missingness").

    Returns:
        masked_batch: dict with the same keys, values zeroed where masked.
        mask: [B, 3] float32 tensor, 1.0 = present, 0.0 = absent, ordered
            (text, audio, vision) to match the gate's expected input order
            (Section 2.3).
    """
    modalities = ("text", "audio", "vision")
    B = batch["text"].shape[0]
    device = batch["text"].device

    keep_prob = 1.0 - rate
    mask = torch.bernoulli(
        torch.full((B, len(modalities)), keep_prob, device=device),
        generator=generator,
    )  # [B, 3], 1 = keep

    # Force-keep one random modality for any sample masked to all-zero.
    all_masked = mask.sum(dim=1) == 0
    if all_masked.any():
        n_all_masked = int(all_masked.sum().item())
        force_idx = torch.randint(
            0, len(modalities), (n_all_masked,), device=device, generator=generator
        )
        mask[all_masked, force_idx] = 1.0

    masked_batch = {}
    for i, name in enumerate(modalities):
        m = mask[:, i].view(B, 1, 1)  # broadcast over [T, D]
        masked_batch[name] = batch[name] * m

    return masked_batch, mask
