"""
Shared modality encoders — Section 2.3.

Each modality (text, audio, vision) is encoded independently by a single-layer,
unidirectional GRU (hidden_size=128), taking the final hidden state as a
fixed-length 128-dim representation per modality per segment. All models in
this study (baselines and proposed method) reuse this identical encoder
design, isolating architectural comparison to the fusion stage.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, hidden_dim] (final hidden state)."""
        _, h_n = self.gru(x)
        return h_n.squeeze(0)  # [B, hidden_dim]


class TriModalEncoders(nn.Module):
    """Convenience wrapper holding one encoder per modality with independent weights."""

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim)
        self.vision_encoder = ModalityEncoder(vision_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "text": self.text_encoder(batch["text"]),
            "audio": self.audio_encoder(batch["audio"]),
            "vision": self.vision_encoder(batch["vision"]),
        }
