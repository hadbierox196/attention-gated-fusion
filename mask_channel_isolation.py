"""
Mask-channel isolation, extended to the dropout-trained gate — manuscript
Section 7, item 3 (previously run only on `gating_only_no_dropout`; see
`supplementary_materials.md` S9 / manuscript Section 3.6).

Methodology, matching Section 3.6 exactly: text features are real
(non-zeroed) throughout. Only the mask value fed to the gate changes, from
[1,1,1] (mask=on, i.e. "text present") to [0,1,1] (mask=off, i.e. "text
marked missing" in the mask channel only). We measure the gate's mean
softmax weight on the text modality under each condition and report the
shift. This isolates the gate's direct sensitivity to its own mask input,
independent of any change in what the encoders actually see.

The original run covered only `gating_only_no_dropout` (never trained with
a varying mask input) to test whether an *untrained* mask channel still
carries signal. This script covers `attention_gated_fusion_full` (trained
with the mask input varying every step, per Section 2.3/2.4) to test the
open question in manuscript Section 4: "we have not run the equivalent
isolation experiment on `attention_gated_fusion_full` and do not want to
claim more than we've directly measured."

Usage:
    python mask_channel_isolation.py \
        --data_path /path/to/aligned_50.pkl \
        --checkpoint_dir /path/to/checkpoints \
        --model attention_gated_fusion_full \
        --seeds 42 123 2024 7 99 \
        --out_csv mask_channel_isolated_effect_dropout_trained.csv

Expects one checkpoint per seed at
    {checkpoint_dir}/{model}_seed{seed}.pt
in the same format `encoder_freeze_swap.py` already reads (a dict with
`text_dim`, `audio_dim`, `vision_dim`, `hidden_dim`, and `encoders`/
`fusion_model` state_dicts) — if your checkpoint naming differs, adjust
`_checkpoint_path` below rather than the rest of the script.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import CmuMosiAligned
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS


def _checkpoint_path(checkpoint_dir: Path, model_name: str, seed: int) -> Path:
    # Real naming convention (confirmed against the actual Drive checkpoints,
    # not assumed): double underscore before "seed", e.g.
    # attention_gated_fusion_full__seed42.pt
    return checkpoint_dir / f"{model_name}__seed{seed}.pt"


def _load(ckpt_path: Path, model_name: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoders = TriModalEncoders(
        text_dim=ckpt["text_dim"], audio_dim=ckpt["audio_dim"],
        vision_dim=ckpt["vision_dim"], hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    fusion_model = MODEL_REGISTRY[model_name](hidden_dim=ckpt["hidden_dim"]).to(device)
    # Real checkpoint keys (confirmed against the actual Drive checkpoints,
    # not assumed): encoders_state_dict / fusion_model_state_dict.
    encoders.load_state_dict(ckpt["encoders_state_dict"])
    fusion_model.load_state_dict(ckpt["fusion_model_state_dict"])
    encoders.eval()
    fusion_model.eval()
    return encoders, fusion_model


@torch.no_grad()
def mean_text_weight_under_mask(encoders, fusion_model, loader, device, text_mask_value: float):
    """text_mask_value: 1.0 for mask=on ([1,1,1]), 0.0 for mask=off ([0,1,1])."""
    total_weight, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings = encoders(batch)  # real, non-zeroed features throughout
        b = batch["text"].shape[0]
        mask = torch.ones(b, 3, device=device)
        mask[:, 0] = text_mask_value  # index 0 = text, per gate.py's (text, audio, vision) ordering
        _, weights = fusion_model(embeddings, mask)  # [B, 3]
        total_weight += weights[:, 0].sum().item()
        n += b
    return total_weight / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model", default="attention_gated_fusion_full",
                         choices=[m for m in GATED_MODELS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 99])
    parser.add_argument("--out_csv", default="mask_channel_isolated_effect_dropout_trained.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    test_ds = CmuMosiAligned(args.data_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    rows = []
    for seed in args.seeds:
        ckpt_path = _checkpoint_path(checkpoint_dir, args.model, seed)
        if not ckpt_path.exists():
            print(f"  [skip] no checkpoint found at {ckpt_path}")
            continue
        encoders, fusion_model = _load(ckpt_path, args.model, device)
        w_on = mean_text_weight_under_mask(encoders, fusion_model, test_loader, device, text_mask_value=1.0)
        w_off = mean_text_weight_under_mask(encoders, fusion_model, test_loader, device, text_mask_value=0.0)
        shift = w_off - w_on
        rows.append({"model": args.model, "seed": seed,
                      "mean_gate_weight_text_mask_on": w_on,
                      "mean_gate_weight_text_mask_off": w_off,
                      "shift": shift})
        print(f"seed={seed}: mask=on {w_on:.4f}  mask=off {w_off:.4f}  shift {shift:+.4f}")

    if not rows:
        print("No checkpoints found — nothing written. Check --checkpoint_dir and naming.")
        return

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print("\nCompare directly against manuscript Section 3.6's gating_only_no_dropout table:")
    print("if shifts here are also <0.01 in every seed, the mask-conditioning mechanism is")
    print("near-inert regardless of dropout training, not just when untrained on that input.")
    print("If shifts are meaningfully larger here, dropout training taught the gate to use")
    print("its mask input after all, and Section 4's mask-inertness claim needs qualifying")
    print("to 'only for the non-dropout-trained gate' rather than 'the gate in general.'")


if __name__ == "__main__":
    main()
