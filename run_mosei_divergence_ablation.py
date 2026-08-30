"""
Controlled ablation to test the two candidate factors named (but not tested)
in manuscript Section 3.7 / Section 7 item 7 for why `gating_only_no_dropout`
avoids collapse on CMU-MOSEI but not CMU-MOSI:

  Factor 1 (training-set size): CMU-MOSEI's train split (N=16,326) is ~13x
    larger than CMU-MOSI's (N=1,284). Ablation: subsample MOSEI's train split
    down to N=1,284 (CMU-MOSI's exact size) before training, keep everything
    else identical, and see whether collapse reappears.

  Factor 2 (base-rate balance): CMU-MOSEI's test-set majority-class rate is
    0.5098 (near-balanced); CMU-MOSI's is 0.596 (more skewed). Ablation:
    resample MOSEI's train split (via majority-class undersampling) so its
    label distribution matches CMU-MOSI's 0.596 majority rate, keep the full
    N=16,326 pre-resampling pool otherwise available, and see whether
    collapse reappears.

Both ablations only touch `gating_only_no_dropout` (the model that actually
diverges) and reuse `train.py`'s existing `train_one_run` unchanged via its
`dataset_cls` parameter, so this script does not reimplement training.

This is a controlled comparison, not a full replication: 5 seeds, one model,
one modality-missing condition (text, matching Section 3.4/3.7's collapse
definition), evaluated at F1<0.05.

Usage:
    python run_mosei_divergence_ablation.py \
        --data_path /path/to/mosei_aligned_50.pkl \
        --ablation trainsize \
        --out_csv mosei_divergence_ablation_trainsize.csv

    python run_mosei_divergence_ablation.py \
        --data_path /path/to/mosei_aligned_50.pkl \
        --ablation baserate \
        --out_csv mosei_divergence_ablation_baserate.csv

Run both. If collapse reappears under `trainsize` but not `baserate`,
training-set size is the more likely factor (and vice versa). If collapse
reappears under neither, the divergence isn't explained by either named
factor and Section 7 item 7 should say so plainly rather than picking one

FOLLOW-UP (manuscript Section 3.7.1's proposed next step, added this
revision): a single 5-seed `trainsize` run showing 1/5 collapses is
directional but not conclusive — it can't distinguish "training-set size
really is the mechanism" from "seed 42's particular random subsample
happened to be unlucky." Use --n_draws to repeat the subsample draw
multiple times per seed, independent of the seed's model-initialization
randomness:

    python run_mosei_divergence_ablation.py \
        --data_path /path/to/mosei_aligned_50.pkl \
        --ablation trainsize \
        --n_draws 5 \
        --out_csv mosei_divergence_ablation_trainsize_multidraw.csv

This runs 5 seeds x 5 draws = 25 total training runs (5x the original
compute for this ablation — budget accordingly). Each draw uses a distinct,
deterministic subsample of MOSEI's train set (draw=0 exactly reproduces the
original single-draw result), while every draw for a given seed starts from
an identically-initialized model, isolating "which particular subsample"
as the only thing varying within a seed. Read the per-seed draw breakdown
the script prints, not just the aggregate collapse rate: if collapse
clusters within specific seeds across most/all of their draws, that is much
stronger evidence for training-set size as the mechanism than if collapse
is thinly scattered across many different (seed, draw) pairs with no
seed dominating.
post hoc. If it reappears under both, they're likely confounded and cannot
be cleanly separated with this design alone.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from data.dataset import binarize
from data.dataset_mosei import CmuMoseiAligned
from missingness import apply_missingness
from models.encoders import TriModalEncoders
from train import MODEL_REGISTRY, set_seed

MODEL_NAME = "gating_only_no_dropout"  # the one model that actually diverges
MOSI_TRAIN_N = 1284       # manuscript Section 2.1: "standard train/valid/test split (1284/229/686)"
MOSI_MAJORITY_RATE = 0.596  # manuscript Table 3.4 / Section 3.7: 409/686


class _SubsetWrapper(Dataset):
    """Thin wrapper exposing the same interface as CmuMoseiAligned/CmuMosiAligned
    (text_dim/audio_dim/vision_dim properties, __getitem__ by index) over a
    fixed list of indices into an underlying full dataset."""

    def __init__(self, base: CmuMoseiAligned, indices: np.ndarray):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

    @property
    def text(self):
        return self.base.text[self.indices]

    @property
    def audio_dim(self):
        return self.base.audio_dim

    @property
    def vision_dim(self):
        return self.base.vision_dim


def _labels_array(ds: CmuMoseiAligned) -> np.ndarray:
    # binarize() matches manuscript Section 2.2's `label > 0` convention.
    return binarize(ds.labels).numpy()


def _make_trainsize_subset(full_train: CmuMoseiAligned, seed: int, target_n: int, draw: int = 0) -> np.ndarray:
    # `draw` lets multiple independent subsamples be drawn for the same
    # training seed, decoupled from the model-training RNG (set_seed(seed)
    # below): draw=0 reproduces the original single-draw behavior exactly
    # (rng seeded by `seed` alone), draw>0 uses a distinct, deterministic
    # RNG stream per draw so repeated runs are reproducible per-draw too.
    rng = np.random.default_rng(seed if draw == 0 else (seed, draw))
    return rng.choice(len(full_train), size=target_n, replace=False)


def _make_baserate_subset(full_train: CmuMoseiAligned, seed: int, target_majority_rate: float) -> np.ndarray:
    """Resample full_train's labels so the majority-class rate matches
    target_majority_rate, keeping as many samples as possible.

    Direction matters and is chosen automatically: if the natural majority
    rate is already below the target (MOSEI's real case: ~0.51 natural vs.
    MOSI's 0.596 target), the majority rate must be RAISED, which means
    undersampling the MINORITY class while keeping every majority-class
    sample. If the natural rate is above target, the majority class itself
    is undersampled instead. An earlier version of this function only
    implemented the second direction, which silently no-ops (returns the
    full unmodified dataset) when the target is above the natural rate —
    confirmed via a real run where majority_rate and n_train both came back
    completely unchanged. This version handles both directions and raises
    if a clip would prevent hitting the target, rather than proceeding
    silently with the wrong achieved rate.
    """
    rng = np.random.default_rng(seed)
    labels = _labels_array(full_train)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_rate = len(pos_idx) / len(labels)
    minority_is_pos = pos_rate < 0.5
    minority_idx, majority_idx = (pos_idx, neg_idx) if minority_is_pos else (neg_idx, pos_idx)
    n_minority, n_majority = len(minority_idx), len(majority_idx)
    natural_majority_rate = n_majority / (n_majority + n_minority)

    if target_majority_rate >= natural_majority_rate:
        # Raise the majority rate: undersample the minority class, keep every majority sample.
        n_minority_target = int(round(n_majority * (1 - target_majority_rate) / target_majority_rate))
        if n_minority_target > n_minority:
            raise ValueError(
                f"Cannot reach target_majority_rate={target_majority_rate} by undersampling the "
                f"minority class alone: would need n_minority_target={n_minority_target} but only "
                f"{n_minority} minority samples exist. Target is unreachable with this pool size."
            )
        sampled_minority = rng.choice(minority_idx, size=n_minority_target, replace=False)
        combined = np.concatenate([sampled_minority, majority_idx])
    else:
        # Lower the majority rate: undersample the majority class, keep every minority sample.
        n_majority_target = int(round(target_majority_rate * n_minority / (1 - target_majority_rate)))
        if n_majority_target > n_majority:
            raise ValueError(
                f"Cannot reach target_majority_rate={target_majority_rate} by undersampling the "
                f"majority class alone: would need n_majority_target={n_majority_target} but only "
                f"{n_majority} majority samples exist. Target is unreachable with this pool size."
            )
        sampled_majority = rng.choice(majority_idx, size=n_majority_target, replace=False)
        combined = np.concatenate([minority_idx, sampled_majority])
    rng.shuffle(combined)
    return combined


def train_and_eval_one_seed(data_path: str, seed: int, ablation: str, device: str, draw: int = 0) -> dict:
    """draw: which subsample draw this is (trainsize ablation only; ignored
    for baserate, which is deterministic given seed). set_seed(seed) is
    called fresh for each draw so every draw trains from an identically
    initialized model — isolating "which particular subsample" as the only
    thing that varies across draws for a fixed seed, rather than conflating
    it with model-init randomness too."""
    set_seed(seed)
    full_train = CmuMoseiAligned(data_path, split="train")
    valid_ds = CmuMoseiAligned(data_path, split="valid")
    test_ds = CmuMoseiAligned(data_path, split="test")

    if ablation == "trainsize":
        idx = _make_trainsize_subset(full_train, seed, MOSI_TRAIN_N, draw=draw)
    elif ablation == "baserate":
        idx = _make_baserate_subset(full_train, seed, MOSI_MAJORITY_RATE)
    else:
        raise ValueError(f"unknown ablation: {ablation}")
    train_ds = _SubsetWrapper(full_train, idx)

    achieved_majority_rate = max(
        _labels_array(full_train)[idx].mean(), 1 - _labels_array(full_train)[idx].mean()
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    encoders = TriModalEncoders(
        text_dim=train_ds.text.shape[-1], audio_dim=train_ds.audio_dim,
        vision_dim=train_ds.vision_dim, hidden_dim=128,
    ).to(device)
    fusion_model = MODEL_REGISTRY[MODEL_NAME](hidden_dim=128).to(device)
    params = list(encoders.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    best_val_f1, best_state = -1.0, None
    for epoch in range(15):
        encoders.train(); fusion_model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            masked_batch, mask = apply_missingness(batch, 0.0)  # gating_only_no_dropout: rate always 0 at train
            embeddings = encoders(masked_batch)
            pred, _ = fusion_model(embeddings, mask)
            loss = loss_fn(pred, batch["label"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        encoders.eval(); fusion_model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                masked_batch, mask = apply_missingness(batch, 0.0)
                embeddings = encoders(masked_batch)
                pred, _ = fusion_model(embeddings, mask)
                preds.append((pred > 0).cpu()); labels.append(binarize(batch["label"]).cpu())
            val_f1 = f1_score(torch.cat(labels).numpy(), torch.cat(preds).numpy())
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {"encoders": {k: v.cpu().clone() for k, v in encoders.state_dict().items()},
                           "fusion_model": {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}}

    encoders.load_state_dict(best_state["encoders"])
    fusion_model.load_state_dict(best_state["fusion_model"])
    encoders.eval(); fusion_model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["text"] = torch.zeros_like(batch["text"])  # text-missing condition, matches Section 3.4/3.7
            mask = torch.zeros(batch["text"].shape[0], 3, device=device); mask[:, 1:] = 1.0
            embeddings = encoders(batch)
            pred, _ = fusion_model(embeddings, mask)
            preds.append((pred > 0).cpu()); labels.append(binarize(batch["label"]).cpu())
        p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        acc, f1 = accuracy_score(l, p), f1_score(l, p)

    return {"seed": seed, "ablation": ablation, "draw": draw, "n_train_used": len(idx),
            "achieved_majority_rate": achieved_majority_rate,
            "text_missing_accuracy": acc, "text_missing_f1": f1,
            "collapsed": f1 < 0.05}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ablation", required=True, choices=["trainsize", "baserate"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 99])
    parser.add_argument("--n_draws", type=int, default=1,
                         help="Only meaningful for --ablation trainsize. Number of independent "
                              "random subsample draws per seed, to separate 'smaller data in "
                              "general' from 'this particular unlucky subsample' (manuscript "
                              "Section 3.7.1's proposed follow-up). Ignored for baserate, which "
                              "is deterministic given seed. n_draws=1 (default) reproduces the "
                              "original single-draw behavior exactly.")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()
    if args.ablation == "baserate" and args.n_draws > 1:
        print("--n_draws > 1 has no effect for --ablation baserate (deterministic given seed); "
              "proceeding with a single run per seed.")
    out_csv = args.out_csv or f"mosei_divergence_ablation_{args.ablation}.csv"

    # Resume support: if out_csv already has some (seed, draw) pairs from a
    # prior, interrupted run, skip them rather than redoing work. This
    # matters here specifically because this script's runs are long enough
    # (25 training runs for a 5-draw trainsize sweep) that a Colab
    # disconnect losing all progress is a real, repeated cost, not a
    # hypothetical one — same lesson as run_mosei_targeted_replication.py's
    # checkpointing (see PROVENANCE.md).
    completed_pairs = set()
    existing_rows = []
    out_path = Path(out_csv)
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            for _, r in existing_df.iterrows():
                completed_pairs.add((int(r["seed"]), int(r.get("draw", 0))))
                existing_rows.append(r.to_dict())
            print(f"Resuming from existing {out_csv}: {len(completed_pairs)} pair(s) already done.")
        except Exception as e:
            print(f"Could not read existing {out_csv} ({e}); starting fresh.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_draws = args.n_draws if args.ablation == "trainsize" else 1
    all_pairs = [(seed, draw) for seed in args.seeds for draw in range(n_draws)]
    pending_pairs = [p for p in all_pairs if p not in completed_pairs]
    total_pairs = len(all_pairs)

    print(f"Total pairs: {total_pairs}  |  already done: {len(completed_pairs)}  |  "
          f"remaining: {len(pending_pairs)}")

    rows = list(existing_rows)
    run_times = []
    for i, (seed, draw) in enumerate(pending_pairs):
        draw_label = f" draw={draw}" if n_draws > 1 else ""
        start = time.time()
        clock = time.strftime("%H:%M:%S")
        print(f"[{clock}] === [{args.ablation}] seed={seed}{draw_label} "
              f"({len(completed_pairs) + i + 1}/{total_pairs} pairs) ===")
        row = train_and_eval_one_seed(args.data_path, seed, args.ablation, device, draw=draw)
        elapsed = time.time() - start
        run_times.append(elapsed)
        avg = sum(run_times) / len(run_times)
        remaining = len(pending_pairs) - (i + 1)
        eta_seconds = int(avg * remaining)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        rows.append(row)
        print(f"  n_train={row['n_train_used']}  majority_rate={row['achieved_majority_rate']:.4f}  "
              f"f1={row['text_missing_f1']:.4f}  collapsed={row['collapsed']}")
        print(f"  done in {elapsed:.0f}s | {len(completed_pairs) + i + 1}/{total_pairs} complete | "
              f"avg {avg:.0f}s/pair | ETA {eta_str} remaining")

        # Flush incrementally so a disconnect after this point loses at most
        # the current in-progress run, not everything done so far.
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n_collapsed = df["collapsed"].sum()
    print(f"\nWrote {out_csv}")
    print(f"Collapse rate under [{args.ablation}] ablation: {n_collapsed}/{len(df)}")
    if n_draws > 1:
        print("\nPer-seed breakdown across draws (this is the check that separates 'smaller")
        print("data in general' from 'this particular unlucky subsample'):")
        per_seed = df.groupby("seed")["collapsed"].agg(["sum", "count"])
        for seed, r in per_seed.iterrows():
            print(f"  seed={seed}: {int(r['sum'])}/{int(r['count'])} draws collapsed")
        print("\nIf collapse clusters in specific seeds across most/all draws, training-set size")
        print("is implicated robustly (not a single unlucky draw). If collapse is scattered thinly")
        print("across many different (seed, draw) pairs with no seed dominating, that's weaker,")
        print("more diffuse evidence — still real, but don't describe it as 'confirmed' either way")
        print("without looking at this per-seed breakdown, not just the aggregate rate.")
    print("\nCompare against: MOSEI full-scale gating_only_no_dropout = 0/5 collapses (Section 3.7),")
    print("MOSI gating_only_no_dropout = 3/5 collapses (Section 3.4).")
    print("If this ablation's collapse rate moves toward MOSI's 3/5, that factor is implicated.")


if __name__ == "__main__":
    main()
