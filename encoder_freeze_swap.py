"""
Encoder freeze/swap test — manuscript Section 7, item 1 (previously unrun).

For each seed, loads the gating_only_no_dropout (ND) and
attention_gated_fusion_full (D) checkpoints, then evaluates all 4
encoder x fusion_model combinations:
    ND-encoders + ND-fusion   (= gating_only_no_dropout itself, sanity baseline)
    D-encoders  + D-fusion    (= attention_gated_fusion_full itself, sanity baseline)
    ND-encoders + D-fusion    (swap: no-dropout encoders, dropout-trained gate)
    D-encoders  + ND-fusion   (swap: dropout-trained encoders, no-dropout gate)

at the 4 standard missingness rates AND on the single-modality "text missing"
(audio+vision-only) condition, which is where manuscript Section 3.4 reports
its 19-point gap. Whichever combination the gap follows tells you whether
it's an encoder effect or a gate effect.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import CmuMosiAligned, binarize
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY

ND_MODEL = "gating_only_no_dropout"
D_MODEL = "attention_gated_fusion_full"
MODALITIES = ("text", "audio", "vision")


def _load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoders = TriModalEncoders(
        text_dim=ckpt["text_dim"], audio_dim=ckpt["audio_dim"],
        vision_dim=ckpt["vision_dim"], hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    encoders.load_state_dict(ckpt["encoders_state_dict"])
    fusion_model = MODEL_REGISTRY[ckpt["model_name"]](hidden_dim=ckpt["hidden_dim"]).to(device)
    fusion_model.load_state_dict(ckpt["fusion_model_state_dict"])
    return encoders, fusion_model


@torch.no_grad()
def _eval_combo(encoders, fusion_model, loader, device, rate, seed_for_masking):
    encoders.eval(); fusion_model.eval()
    gen = torch.Generator(device=device).manual_seed(seed_for_masking)
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_batch, mask = apply_missingness(batch, rate, generator=gen)
        embeddings = encoders(masked_batch)
        pred, _ = fusion_model(embeddings, mask)
        preds.append((pred > 0).cpu())
        labels.append(binarize(batch["label"]).cpu())
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"accuracy": accuracy_score(l, p), "f1": f1_score(l, p), "n": len(l)}


@torch.no_grad()
def _eval_text_missing(encoders, fusion_model, loader, device):
    """Single-modality masking: text absent, audio+vision present — Section 3.4's condition."""
    encoders.eval(); fusion_model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["text"].shape[0]
        mask = torch.ones(B, 3, device=device)
        mask[:, 0] = 0.0  # text absent
        masked_batch = dict(batch)
        masked_batch["text"] = batch["text"] * 0.0
        embeddings = encoders(masked_batch)
        pred, _ = fusion_model(embeddings, mask)
        preds.append((pred > 0).cpu())
        labels.append(binarize(batch["label"]).cpu())
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"accuracy": accuracy_score(l, p), "f1": f1_score(l, p), "n": len(l)}


def run(config_data_path, checkpoint_dir, seeds, batch_size=32, eval_rates=(0.0, 0.25, 0.5, 0.75)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(checkpoint_dir)
    test_ds = CmuMosiAligned(config_data_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    rows = []
    for seed in seeds:
        nd_ckpt = checkpoint_dir / f"{ND_MODEL}__seed{seed}.pt"
        d_ckpt = checkpoint_dir / f"{D_MODEL}__seed{seed}.pt"
        if not (nd_ckpt.exists() and d_ckpt.exists()):
            print(f"  seed={seed}: missing checkpoint(s), skipping"); continue

        enc_nd, fus_nd = _load(nd_ckpt, device)
        enc_d, fus_d = _load(d_ckpt, device)

        combos = {
            "ND_enc__ND_gate": (enc_nd, fus_nd),
            "D_enc__D_gate": (enc_d, fus_d),
            "ND_enc__D_gate_SWAP": (enc_nd, fus_d),
            "D_enc__ND_gate_SWAP": (enc_d, fus_nd),
        }

        for combo_name, (enc, fus) in combos.items():
            for rate in eval_rates:
                masking_seed = seed * 1000 + int(rate * 100)
                m = _eval_combo(enc, fus, test_loader, device, rate, masking_seed)
                rows.append({"seed": seed, "combo": combo_name, "condition": f"rate_{rate}", **m})
            m_text = _eval_text_missing(enc, fus, test_loader, device)
            rows.append({"seed": seed, "combo": combo_name, "condition": "text_missing", **m_text})
            print(f"seed={seed} {combo_name}: text_missing acc={m_text['accuracy']:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 99])
    parser.add_argument("--out_csv", default="encoder_freeze_swap_results.csv")
    args = parser.parse_args()

    df = run(args.data_path, args.checkpoint_dir, args.seeds)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv} ({len(df)} rows)")

    print("\n=== text_missing accuracy by combo (mean across seeds) ===")
    print(df[df.condition == "text_missing"].groupby("combo")["accuracy"].agg(["mean", "std"]).to_string())
