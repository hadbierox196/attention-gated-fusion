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
                   dropout_rate=None, verbose_masked_collisions=True,
                   dataset_cls=None,
                   device="cuda" if torch.cuda.is_available() else "cpu"):
    if dataset_cls is None:
        dataset_cls = CmuMosiAligned
    set_seed(seed)
    ds = dataset_cls(data_path, split="train")
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

    # Fix #1 (manuscript Section 2.6 / checklist item #1): sample the
    # missingness rate ONCE before the loop, then re-apply it to build the
    # SAME masked batch every step, instead of resampling inside the loop
    # every one of the 50 steps.
    #
    # Fix #2 (found after Fix #1 alone did not make the check pass): if
    # dropout_rate is left as None, the rate is still drawn from a shared
    # global RNG stream AFTER encoders/fusion_model are constructed. Since
    # each architecture initializes a different number of parameters, the
    # RNG stream has drifted by a different amount by the time this draw
    # happens — so different models land on different, uncontrolled rates
    # purely as a side effect of their parameter count, not by design. Pass
    # --dropout_rate explicitly to remove this confound and test every
    # dropout-trained model at the same, chosen rate.
    if dropout_rate is not None:
        rate = float(dropout_rate) if use_dropout else 0.0
        rate_source = "explicit"
    else:
        rate = float(torch.empty(1).uniform_(0.0, 0.75).item()) if use_dropout else 0.0
        rate_source = "RNG-drawn (uncontrolled — pass --dropout_rate to fix)"
    masked_batch, mask = apply_missingness(batch, rate)

    losses = []
    for step in range(steps):
        embeddings = encoders(masked_batch)
        out = fusion_model(embeddings, mask)
        pred = out[0] if is_gated else out
        loss = loss_fn(pred, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[{model_name}] rate={rate:.3f} ({rate_source}, fixed for all {steps} steps)  "
          f"step 0 loss={losses[0]:.4f}  step {steps-1} loss={losses[-1]:.4f}")

    if verbose_masked_collisions and rate > 0.0:
        # Diagnostic only, no effect on training. Checks whether the fixed
        # missingness mask makes any of the n_samples indistinguishable to
        # the model (identical embeddings) while having different labels --
        # if so, a nonzero loss floor is expected and correct, not a bug.
        with torch.no_grad():
            final_embeddings = encoders(masked_batch)
            concat = torch.cat(
                [final_embeddings["text"], final_embeddings["audio"], final_embeddings["vision"]],
                dim=-1,
            )
            labels = batch["label"]
            n = concat.shape[0]
            collisions = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = torch.norm(concat[i] - concat[j]).item()
                    label_gap = abs(labels[i].item() - labels[j].item())
                    if dist < 1e-3 and label_gap > 0.5:
                        collisions.append((i, j, dist, label_gap))
            if collisions:
                print(f"    [collision check] {len(collisions)} sample pair(s) have "
                      f"near-identical masked embeddings but different labels "
                      f"(e.g. samples {collisions[0][0]},{collisions[0][1]}: "
                      f"embedding dist={collisions[0][2]:.5f}, label gap={collisions[0][3]:.3f}) "
                      f"-- a nonzero loss floor is expected here, not necessarily a training bug.")
            else:
                print(f"    [collision check] no near-identical-embedding/different-label "
                      f"pairs found among the {n} samples -- a persistent nonzero loss floor "
                      f"is NOT explained by this, and points toward a genuine training issue.")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=None,
                         help="Fix the missingness rate used for dropout-trained models "
                              "(applies to attention_gated_fusion_full, dropout_only_fusion, "
                              "hard_mask_gated_fusion) instead of letting it be drawn from an "
                              "RNG stream whose state has drifted per-model. Non-dropout models "
                              "always use rate=0.0 regardless of this flag.")
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi",
                         help="Which dataset's loader to use. 'mosei' requires "
                              "data/dataset_mosei.py's FEATURE_RELEASE_NOTE to be "
                              "filled in first (checklist item #1/#8).")
    args = parser.parse_args()

    if args.dataset == "mosei":
        from data.dataset_mosei import CmuMoseiAligned as _DatasetCls
    else:
        _DatasetCls = CmuMosiAligned

    results = {m: overfit_check(m, args.data_path, n_samples=args.n_samples, steps=args.steps,
                                 dropout_rate=args.dropout_rate, dataset_cls=_DatasetCls)
               for m in args.models}

    print("\n=== Summary ===")
    all_ok = True
    for m, losses in results.items():
        ok = losses[-1] < 0.05 * losses[0]
        all_ok &= ok
        print(f"{m}: {'PASS' if ok else 'CHECK MANUALLY'} (loss {losses[0]:.4f} -> {losses[-1]:.4f})")
    if not all_ok:
        raise SystemExit("One or more models did not overfit as expected — inspect before trusting the full grid.")
