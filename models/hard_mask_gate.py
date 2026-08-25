"""
Hard-mask attention gate — implements Future Work item #2 from the manuscript
(Section 7): "architecturally excluding masked features from the gate's
softmax rather than relying on soft mask-conditioning."

Where AttentionGate (gate.py) concatenates the mask as an *input signal* the
gate must learn to use, HardMaskAttentionGate structurally guarantees a
masked-out modality receives exactly zero fusion weight, by additively
masking its logit to -inf before the softmax. This directly targets the
diagnostic finding in Section 4: mask-conditioning alone (soft) is not
sufficient to override a learned informativeness prior, so this variant
removes the model's ability to ignore the mask at all.

STATUS: implemented but not yet run against real data or compared to
attention_gated_fusion_full — see manuscript Section 7. This is provided as
a ready-to-run addition to the experiment grid, not as a verified result.
To evaluate it, add "hard_mask_gated_fusion" to config.yaml's `models` list
and register it in train.py's MODEL_REGISTRY / GATED_MODELS.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .gate import AttentionGate


class HardMaskAttentionGate(AttentionGate):
    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=-1
        )
        gate_input = torch.cat([concat, mask], dim=-1)
        logits = self.net(gate_input)  # [B, 3]

        # Additively mask logits for absent modalities before softmax, so a
        # masked-out modality structurally cannot receive nonzero weight —
        # regardless of what the learned network would otherwise assign it.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(mask == 0, neg_inf)

        weights = torch.softmax(masked_logits, dim=-1)
        return weights


class HardMaskGatedFusion(nn.Module):
    """Full model using HardMaskAttentionGate — parallel structure to AttentionGatedFusion."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate = HardMaskAttentionGate(hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        weights = self.gate(embeddings, mask)
        stacked = torch.stack(
            [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=1
        )
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        pred = self.head(fused).squeeze(-1)
        return pred, weights
