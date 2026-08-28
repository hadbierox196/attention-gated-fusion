"""
Training-pipeline sanity check (manuscript Section 2.6).
Overfits a small fixed subset to near-zero loss to confirm the forward/
backward pass is wired correctly, before trusting a full multi-seed grid.
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import CmuMosiAligned
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, GATED_MODELS, DROPOUT_TRAINED, set_seed


def overfit_check(model_name, data_path, n_samples=16, steps=50, lr=1e-3, seed=42,
                   device="cuda" if torch.cuda.is_available() else "cpu"):
    set_seed(seed)
    ds = CmuMosiAligned(data_path, split="train")
    subset = Subset(ds, list(range(n_samples)))
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    encoders = TriModalEncoders(
        text_dim=ds.text.shape[-1], audio_dim=ds.audio_dim,
        vision_dim=ds.vision_dim, hidden_dim=128,
    ).to(device)
    fusion_model = MODEL_REGISTRY[model_name](hidden_dim=128).to(device)
    is_gated = model_name in GATED_MODELS
    use_dropout = model_name in DROPOUT_TRAINED

    params = list(encoders.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for step in range(steps):
        rate = float(torch.empty(1).uniform_(0.0, 0.75).item()) if use_dropout else 0.0
        masked_batch, mask = apply_missingness(batch, rate)
        embeddings = encoders(masked_batch)
        out = fusion_model(embeddings, mask)
        pred = out[0] if is_gated else out
        loss = loss_fn(pred, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[{model_name}] step 0 loss={losses[0]:.4f}  step {steps-1} loss={losses[-1]:.4f}")
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    results = {m: overfit_check(m, args.data_path, n_samples=args.n_samples, steps=args.steps)
               for m in args.models}

    print("\n=== Summary ===")
    all_ok = True
    for m, losses in results.items():
        ok = losses[-1] < 0.05 * losses[0]
        all_ok &= ok
        print(f"{m}: {'PASS' if ok else 'CHECK MANUALLY'} (loss {losses[0]:.4f} -> {losses[-1]:.4f})")
    if not all_ok:
        raise SystemExit("One or more models did not overfit as expected — inspect before trusting the full grid.")
