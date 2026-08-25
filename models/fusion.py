"""
Baseline fusion variants — Section 2.3 baseline table.

All baselines share the TriModalEncoders from encoders.py; only the fusion
mechanism differs, per the manuscript's explicit design goal of isolating
architectural comparison to the fusion stage.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """Concatenate all three embeddings, feed through MLP head."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=-1
        )
        return self.head(concat).squeeze(-1)


class LateFusion(nn.Module):
    """Independent linear head per modality, average predictions."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.text_head = nn.Linear(hidden_dim, 1)
        self.audio_head = nn.Linear(hidden_dim, 1)
        self.vision_head = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        preds = torch.stack(
            [
                self.text_head(embeddings["text"]).squeeze(-1),
                self.audio_head(embeddings["audio"]).squeeze(-1),
                self.vision_head(embeddings["vision"]).squeeze(-1),
            ],
            dim=1,
        )  # [B, 3]
        return preds.mean(dim=1)


class FixedWeightFusion(nn.Module):
    """Concatenate embeddings weighted by a fixed prior (0.4 text, 0.3 audio, 0.3 vision)."""

    def __init__(self, hidden_dim: int = 128, prior=(0.4, 0.3, 0.3)):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior).view(1, 3, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=1
        )  # [B, 3, hidden_dim]
        fused = (self.prior * stacked).sum(dim=1)  # [B, hidden_dim]
        return self.head(fused).squeeze(-1)


class DropoutOnlyFusion(EarlyFusion):
    """
    Identical architecture to EarlyFusion; distinguished only by training
    regime (trained WITH modality-dropout augmentation via apply_missingness
    at train time). No architectural difference from EarlyFusion — see
    train.py for where the dropout-training distinction actually lives.
    """


class ImputationBaselinePost2023(nn.Module):
    """
    Representative reimplementation of the DAST-GAN-family reconstruction-
    before-fusion principle (Section 1.3, Section 2.3) — NOT a reproduction
    of DAST-GAN's full architecture. Concatenated embeddings are passed
    through a reconstruction sub-network that imputes masked-modality
    embeddings from available ones before fusion.

    NOTE: this is a minimal stand-in. Before treating this module's output
    as matching the manuscript's Table 3.1/3.3 numbers for
    `imputation_baseline_post2023`, verify against whatever reconstruction
    architecture actually produced those numbers — the +58% parameter count
    reported in Section 3.3 implies a specific sub-network size that this
    stub does not attempt to match exactly.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Reconstruction sub-network: predicts all 3 modality embeddings from
        # the concatenated (masked) input + mask, then fuses the reconstructed set.
        self.reconstruct = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=-1
        )
        recon_input = torch.cat([concat, mask], dim=-1)
        reconstructed = self.reconstruct(recon_input)  # [B, 3*hidden_dim]
        return self.head(reconstructed).squeeze(-1)
