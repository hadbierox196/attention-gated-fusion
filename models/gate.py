"""
Attention-gated fusion (proposed method) — Section 2.3.

gate(concat_features, mask) = Softmax(Linear(64) -> ReLU -> Linear(3))

The three 128-dim modality embeddings are concatenated (384-dim) and further
concatenated with the 3-dim binary missingness mask (387-dim total). The gate
takes the mask as an explicit input, not inferred purely from feature
statistics — this is the architectural distinction from prior gated-fusion
work (Section 1.3) that Section 4/5 shows is only partially sufficient.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, hidden_dim: int = 128, gate_hidden: int = 64, n_modalities: int = 3):
        super().__init__()
        in_dim = hidden_dim * n_modalities + n_modalities  # 384 + 3 = 387
        self.net = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_modalities),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: dict of 'text'/'audio'/'vision' -> [B, hidden_dim]
        mask: [B, 3] in (text, audio, vision) order
        returns: [B, 3] softmax gate weights, same modality order as mask
        """
        concat = torch.cat(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=-1
        )  # [B, 384]
        gate_input = torch.cat([concat, mask], dim=-1)  # [B, 387]
        logits = self.net(gate_input)  # [B, 3]
        weights = torch.softmax(logits, dim=-1)
        return weights


class AttentionGatedFusion(nn.Module):
    """Full proposed-method fusion + prediction head (Section 2.3)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate = AttentionGate(hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        weights = self.gate(embeddings, mask)  # [B, 3]
        stacked = torch.stack(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=1
        )  # [B, 3, hidden_dim]
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, hidden_dim]
        pred = self.head(fused).squeeze(-1)  # [B]
        return pred, weights
