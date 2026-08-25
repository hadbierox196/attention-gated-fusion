# Attention-Gated Fusion for Modality-Robust Affective Computing

---

## Abstract

Multimodal emotion and sentiment recognition systems typically assume all modalities are available at inference time, an assumption that frequently fails in practice. Existing solutions either reconstruct missing modalities via auxiliary generative networks, introducing a second failure mode, or train separate models per modality combination, which scales poorly. We propose an attention-gated fusion layer conditioned explicitly on a missingness indicator, trained jointly with modality dropout, that reweights available modalities without reconstructing missing ones. On CMU-MOSI across three seeds, this gate yields a small but statistically consistent improvement over dropout-trained fusion without gating (p<0.05 at missingness rates ≥ 0.25), and a parameter-count check rules out added capacity as the source of this gain. However, diagnostic analysis reveals the gate under-compensates for a missing dominant modality (text): it responds to the missingness mask but never demotes a masked-out modality below others that are genuinely present, causing dropout-trained fusion — gated or not — to underperform static fixed-weight fusion at high missingness rates. We report both results, and identify the gate's insufficient mask-conditioning as a concrete target for future architectural refinement.

---

## 1. Introduction

Multimodal emotion and sentiment recognition systems typically assume all modalities — text, audio, and visual signal — are available at inference time. In practice this assumption frequently breaks: a microphone drops out, a camera is occluded, a transcript is unavailable. Model performance under this kind of partial, unpredictable modality loss is a distinct problem from performance on complete inputs, and one that standard multimodal fusion architectures are not designed to handle gracefully.

Existing approaches to this problem fall into two families, neither of which fully addresses it. **Imputation-based methods** (e.g., MMIN; TFR-Net) reconstruct a missing modality's representation from the modalities that remain present, typically via an auxiliary generative or adversarial sub-network trained to approximate the missing signal. This adds a second failure mode on top of the original task: the model must now also learn a good imputation function, and errors in that reconstruction propagate into the downstream prediction. **Per-combination methods**, which train or select a separate model for each possible subset of available modalities, avoid the imputation problem but scale combinatorially with the number of modalities and are impractical beyond a small fixed set.

A third, lighter-weight approach — training a single fusion model with modality dropout, so it learns to cope with missing inputs during training rather than reconstructing them — is more practical but treats all missingness patterns identically once dropout is applied, without an explicit mechanism for reweighting *which* available modalities to trust more when others are absent.

We investigate whether an **attention-gated fusion layer, conditioned explicitly on a missingness indicator**, can close this gap: a lightweight gate that learns to reweight available modalities based on which are present, trained jointly with modality dropout rather than requiring a separate reconstruction network or per-combination model.

### 1.1 Contribution

We propose an attention-gated fusion layer conditioned explicitly on a missingness indicator, trained with modality dropout, that reweights available modalities without reconstructing missing ones. Across three seeds on CMU-MOSI, this gate yields a small but statistically consistent improvement over dropout-trained fusion without gating (p<0.05 at missingness rates ≥ 0.25). This is a narrower claim than it might first appear: dropout-trained fusion in general — gated or not — does not surpass static fixed-weight fusion at high missingness rates on this dataset, and our own diagnostic analysis (Sections 4–5) shows the gate under-compensates for a missing dominant modality rather than fully redistributing weight to the modalities that remain. We report this alongside the positive result rather than omitting it.

### 1.2 Scope

This work evaluates a single dataset (CMU-MOSI), three modalities (text, audio, vision) with GRU-based encoders, and missingness rates from 0–75%, simulated by zeroing modality features at a controllable per-sample rate. We do not claim state-of-the-art performance across all missingness regimes or modality combinations, and we do not claim the gating mechanism's benefit generalizes beyond dropout-trained fusion specifically — it does not outperform non-dropout-trained static baselines at high missingness in our experiments. We do not claim cross-domain generalization beyond CMU-MOSI; any such claim would require a second-dataset experiment, which is out of scope here and left to future work (Section 7).

### 1.3 Related work

- **MMIN** [CITE] learns to impute missing-modality features via cascaded residual autoencoders trained with cyclic reconstruction loss; unlike our approach, it requires training an auxiliary generative sub-network and inherits that network's own failure modes on out-of-distribution missingness patterns.
- **TFR-Net** [CITE] uses a transformer-based feature reconstruction module for both random and structured (temporal-span) missingness, again relying on learned reconstruction rather than reweighting available signal.
- **Gated multimodal fusion** more broadly [CITE, e.g. Gated Multimodal Units] uses learned gates to combine modalities, but typically conditions the gate on feature statistics alone rather than an explicit missingness indicator — our gate differs in taking the mask as a direct conditioning input, which we show (Section 5) is only partially sufficient to override a learned modality-informativeness prior.
- **Post-2023 missing-modality method** [CITE — primary SOTA comparison]: the `imputation_baseline_post2023` model in our experiments stands in architecturally for this family; pair with your chosen paper's specific method description here.

---

## 2. Methods

### 2.1 Dataset

We use **CMU-MOSI** [CITE Zadeh et al.], a standard benchmark for multimodal sentiment analysis consisting of 2,199 opinion segments from online movie-review videos, each labeled with a continuous sentiment score in [-3, +3]. We binarize this into positive/negative sentiment for classification metrics (accuracy, F1), consistent with prior work in the missing-modality literature.

- **Access:** aligned, pre-extracted features (`aligned_50.pkl`) released by the Self-MM authors [CITE Yu et al.] via Google Drive, derived from the original CMU-MultimodalSDK pipeline.
- **License/access terms:** research use, distributed by the original authors under the same terms as CMU-MultimodalSDK; no additional agreement required beyond standard academic attribution.
- **Splits:** standard train/valid/test split as provided in the aligned feature release.
- **Modalities:** text (GloVe-based, aligned), audio (COVAREP acoustic features), vision (Facet-based visual features), all pre-aligned to a common 50-timestep sequence length per segment.

### 2.2 Preprocessing

- Features loaded directly from the pre-aligned `.pkl` release; no additional alignment step was needed since CMU-MOSI's word-level alignment was already applied upstream.
- Audio and vision features contain occasional `NaN` values from the original extraction pipeline (a known property of COVAREP/Facet output on this dataset); these are replaced with 0 via `np.nan_to_num` before tensor conversion.
- Text, audio, and vision are each converted to `float32` tensors of shape `[N, 50, D]`, where `D` is 300 (text/GloVe), and dataset-native dimensionality for audio and vision respectively (loaded and printed at runtime rather than hardcoded, to catch feature-release version mismatches).
- Regression labels are cast to `float32`; binary labels are derived at evaluation time via `label > 0`.
- No additional normalization or scaling was applied beyond what the original feature release provides.

### 2.3 Model architecture

**Shared modality encoders.** Each modality (text, audio, vision) is encoded independently by a single-layer, unidirectional GRU (`hidden_size=128`), taking the final hidden state as a fixed-length 128-dim representation per modality per segment. All models in this study (baselines and proposed method) reuse this identical encoder design, isolating architectural comparison to the fusion stage.

**Attention-gated fusion (proposed method).** The three 128-dim modality embeddings are concatenated (384-dim) and further concatenated with the 3-dim binary missingness mask (387-dim total), which is passed through a small feedforward gate network:

```
gate(concat_features, mask) = Softmax(Linear(64) -> ReLU -> Linear(3))
```

producing three scalar weights `w_text, w_audio, w_vision` that sum to 1. The fused representation is the weighted sum of the three modality embeddings:

```
fused = w_text * h_text + w_audio * h_audio + w_vision * h_vision
```

followed by a 2-layer MLP head (128 → 128 → 1) predicting sentiment score. Critically, the gate takes the missingness mask as an explicit input (not inferred purely from feature statistics), distinguishing this from prior gated-fusion approaches that condition only on learned features.

**Baselines** (all share the same GRU encoders described above):

| Baseline | Fusion mechanism |
|---|---|
| Early fusion | Concatenate all three embeddings, feed through MLP head |
| Late fusion | Independent linear head per modality, average predictions |
| Fixed-weight fusion | Concatenate embeddings weighted by a fixed prior (0.4 text, 0.3 audio, 0.3 vision) before the MLP head |
| Dropout-only fusion | Identical architecture to early fusion, but trained with modality-dropout augmentation (isolates the training-regime contribution from the gating-architecture contribution) |
| Gating-only, no dropout | Identical architecture to the proposed method, trained *without* dropout augmentation (isolates the architecture contribution from the training-regime contribution) |
| Imputation baseline (post-2023 comparison) | Concatenated embeddings passed through a reconstruction sub-network that imputes masked-modality embeddings from available ones before fusion, standing in for the MMIN/TFR-Net family as the required recent comparison |

### 2.4 Missingness simulation protocol

A single function, `apply_missingness(batch, rate)`, is used identically at train and test time to avoid any train/test protocol mismatch:

- For each sample in a batch, a binary mask over the 3 modalities is drawn such that each modality is independently retained with probability `(1 - rate)`.
- If a draw would zero out all three modalities for a sample, one modality is force-kept at random, guaranteeing at least one modality is always present.
- Masked-out modalities have their entire feature tensor zeroed (not removed from the batch — sequence length and tensor shape are preserved, only the values are zeroed), consistent with how missingness is simulated in the prior gated-fusion and imputation literature we compare against.

**Training-time missingness:** For dropout-trained models, the missingness rate is resampled per training batch from `Uniform(0, 0.75)`, matching the range of rates used at evaluation. (An earlier version of this pipeline used a fixed training-time rate of 0.3, which produced a train/eval distribution mismatch; this was identified and corrected — see Section 5 for the effect this had on results.)

**Evaluation-time missingness:** Models are evaluated at four fixed rates — {0.0, 0.25, 0.5, 0.75} — with the same `apply_missingness` function, seeded identically per model/seed combination so that all models see the same masking pattern at a given rate for a fair comparison.

### 2.5 Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 15 |
| Hidden dimension | 128 (all encoders and gate) |
| Hardware | Single Colab T4 GPU |
| Random seeds | 42, 123, 2024 (3 seeds; see Section 6) |
| Checkpointing | Best validation F1 checkpoint saved per (model, seed) run |
| Loss | MSE on continuous sentiment score; accuracy/F1 computed post-hoc via sign threshold |

All random seeds (Python, NumPy, PyTorch, CUDA) are fixed at the start of each run via a single `set_seed()` call, and re-applied before each evaluation pass to ensure masking patterns are reproducible across model comparisons at a given rate.

### 2.6 Sanity checks

Before running the full experiment grid, the proposed model was verified to overfit a 16-sample subset of the training set to near-zero loss within 50 steps, confirming the training pipeline (data loading, forward pass, loss, backward pass) is functioning correctly prior to full-scale runs.

### 2.7 Reproducibility

The full experiment grid (7 models × 3 seeds × 4 missingness rates = 84 evaluation points from 21 trained models) is generated by re-runnable Colab cells with no manual intervention beyond a one-time dataset download step (fetching `aligned_50.pkl`, documented with a fallback path via personal Google Drive if the shared-file quota is exhausted). All reported tables are produced directly from `results_raw.csv`, written by the experiment-grid cell, with per-run configuration logged alongside in `config_log.json`.

---

## 3. Results

### 3.1 Main metric table (accuracy, mean ± std across 3 seeds)

| Model | 0.00 | 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|
| attention_gated_fusion_full | 0.772 ± 0.009 | 0.678 ± 0.004 | 0.577 ± 0.003 | 0.529 ± 0.003 |
| dropout_only_fusion | 0.760 ± 0.007 | 0.667 ± 0.007 | 0.567 ± 0.001 | 0.519 ± 0.004 |
| gating_only_no_dropout | 0.759 ± 0.006 | 0.728 ± 0.005 | 0.691 ± 0.003 | 0.650 ± 0.007 |
| fixed_weight_fusion | 0.762 ± 0.013 | 0.734 ± 0.011 | 0.693 ± 0.008 | 0.654 ± 0.012 |
| early_fusion | 0.779 ± 0.007 | 0.724 ± 0.029 | 0.658 ± 0.064 | 0.618 ± 0.069 |
| late_fusion | 0.779 ± 0.010 | 0.723 ± 0.046 | 0.659 ± 0.075 | 0.618 ± 0.076 |
| imputation_baseline_post2023 | 0.776 ± 0.016 | 0.685 ± 0.009 | 0.582 ± 0.004 | 0.535 ± 0.004 |

### 3.2 Statistical significance (paired t-test, n=3 seeds — see Section 6)

**attention_gated_fusion_full vs. dropout_only_fusion:**

| Missing rate | t-stat | p-value |
|---|---|---|
| 0.00 | 4.15 | 0.053 |
| 0.25 | 19.47 | 0.003 |
| 0.50 | 7.42 | 0.018 |
| 0.75 | 7.37 | 0.018 |

Gating improves over dropout-only training at every rate ≥ 0.25 (p<0.05); borderline at rate=0.00.

**gating_only_no_dropout vs. fixed_weight_fusion:**

| Missing rate | t-stat | p-value |
|---|---|---|
| 0.00 | 0.84 | 0.49 |
| 0.25 | 0.26 | 0.82 |
| 0.50 | 0.47 | 0.68 |
| 0.75 | -0.24 | 0.83 |

No significant difference at any rate — a learned gate without dropout training is statistically indistinguishable from a fixed (0.4, 0.3, 0.3) weighting.

### 3.3 Efficiency table

| Model | Params | FLOPs | Latency (ms, batch=1, GPU) |
|---|---|---|---|
| early_fusion | 503,681 | 22.9M | 1.81 |
| late_fusion | 454,659 | 22.8M | 2.09 |
| fixed_weight_fusion | 503,681 | 22.9M | 2.13 |
| dropout_only_fusion | 503,681 | 22.9M | 1.91 |
| gating_only_no_dropout | 495,940 | 22.9M | 2.20 |
| imputation_baseline_post2023 | 799,361 | 23.2M | 1.83 |
| attention_gated_fusion_full | 495,940 | 22.9M | 2.32 |

All models share the same GRU encoders, so params/FLOPs are similar throughout; the imputation baseline is the notable outlier at +58% params due to its reconstruction sub-network. Latency differences across all models are small (<0.6ms) and not a practical bottleneck at this scale.

### 3.4 Breakdown by missingness pattern (attention_gated_fusion_full, single modality dropped)

| Missing modality | Accuracy |
|---|---|
| text | 0.404 |
| audio | 0.776 |
| vision | 0.776 |

Text dominates prediction on CMU-MOSI; dropping it alone collapses accuracy near chance, while dropping audio or vision alone is nearly costless.

### 3.5 Gate weight vs. mask sensitivity check

Mean gate weight assigned to text, by whether text is present in the input:

| Rate | Text present | w(text) | w(audio) | w(vision) |
|---|---|---|---|---|
| 0.25 | absent | 0.452 | 0.294 | 0.254 |
| 0.25 | present | 0.583 | 0.232 | 0.186 |
| 0.50 | absent | 0.450 | 0.295 | 0.255 |
| 0.50 | present | 0.576 | 0.235 | 0.189 |
| 0.75 | absent | 0.448 | 0.295 | 0.257 |
| 0.75 | present | 0.566 | 0.239 | 0.196 |

The gate responds to the mask (text weight drops ~0.58 → ~0.45 when text is absent) but never demotes text below audio or vision, even when text is fully zeroed out.

### 3.6 Failure case example (rate=0.5)

| True | Pred | Gate weights (t, a, v) | Mask (t, a, v) |
|---|---|---|---|
| 0.02 | 1.32 | 0.610, 0.258, 0.132 | 0, 1, 1 |
| -2.60 | 0.26 | 0.455, 0.293, 0.252 | 0, 1, 1 |
| -1.80 | 0.26 | 0.455, 0.293, 0.252 | 0, 1, 1 |
| -0.80 | 0.26 | 0.455, 0.293, 0.252 | 0, 1, 1 |
| -2.40 | 0.26 | 0.446, 0.297, 0.257 | 0, 1, 0 |

In every failure case shown, text is the missing modality and the gate still assigns it the largest single weight, illustrating the under-compensation pattern quantified in Section 3.5.

---

## 4. Discussion

**Interpretation of the ablation.** The gap between `attention_gated_fusion_full` and `dropout_only_fusion` is small (+0.010 to +0.012 accuracy) but statistically consistent (p<0.05 at rates ≥ 0.25), confirming the gating mechanism contributes beyond dropout training alone. However, the gap between `gating_only_no_dropout` and `fixed_weight_fusion` is not significant at any rate — a learned gate without dropout exposure adds nothing measurable over a hand-picked static weighting.

**Revisiting the novelty claim.** The gate mechanism is measurably responsive to the missingness mask: mean weight on text drops from 0.58 to 0.45 when text is masked out, consistently across all tested rates. This rules out the possibility that the gate is ignoring its mask input entirely. However, the magnitude of this adjustment is insufficient — masked text retains a higher average weight (0.45) than either audio or vision receives even when those modalities *are* present (0.29 and 0.25–0.26, respectively). Given that text is by far the most predictive modality on CMU-MOSI (dropping it alone reduces accuracy from 0.772 to 0.404, versus <0.01 drop for audio or vision alone), the gate appears to have learned a strong prior toward text informativeness that mask-conditioning only partially overrides.

This explains both headline results: gating's improvement over dropout-only training (the gate does shift weight away from text when absent, which dropout-only fusion cannot do at all) and dropout-trained models' underperformance relative to fixed-weight fusion at high missingness (residual over-reliance on text means the gate fails to fully exploit remaining audio/vision signal, whereas fixed-weight fusion guarantees each modality a floor of 0.3 weight regardless of the gate's learned preferences).

**Alternative explanation considered.** Could the gap between dropout-only and full-method come from added parameters/capacity rather than the gating mechanism specifically? `attention_gated_fusion_full` has 495,940 params versus `dropout_only_fusion`'s 503,681 — the gated model has *fewer* parameters, ruling out a capacity-based explanation for its improvement.

---

## 5. Limitations

- **Single dataset.** All results are on CMU-MOSI (2,199 segments); no cross-dataset validation was run.
- **Seed count.** Statistical significance tests use n=3 seeds (42, 123, 2024), giving 2 degrees of freedom for paired t-tests. This is sufficient to detect the large, consistent effects reported here, but gives low power for smaller or noisier effects, and single-seed outliers could shift borderline results (e.g., the p=0.053 case at rate=0.00). Five or more seeds would substantially strengthen these claims.
- **Modality types.** Only text, audio, and vision as provided by the CMU-MOSI feature release; no raw-signal or alternative feature extraction was tested.
- **Missingness protocol.** Training sampled missingness rate uniformly over (0, 0.75) per batch; we did not test narrower ranges or curriculum scheduling, both of which are plausible mitigations for the text-under-compensation finding (Section 4).
- **Compute budget.** All experiments ran on a single Colab T4 GPU session; no large-scale hyperparameter search was performed (fixed lr=1e-3, hidden=128, 15 epochs).
- **Gate architecture.** The gate uses a soft, mask-concatenated conditioning signal (mask appended to features, then softmax). We did not test harder conditioning (e.g., architecturally excluding masked features from the gate's input) — see Section 7.

---

## 6. Conclusion

We proposed an attention-gated fusion layer conditioned on a missingness mask for modality-robust affective computing. Across three seeds on CMU-MOSI, the gate provides a small, statistically consistent improvement over dropout-trained fusion without gating (p<0.05 at missingness rates ≥ 0.25), and parameter counts rule out capacity as an alternative explanation for this gap. However, this contribution is narrower than the framing in our original problem statement: dropout-trained fusion — gated or not — does not surpass static fixed-weight fusion at high missingness on this dataset, and diagnostic analysis shows the gate under-compensates for a missing dominant modality (text) rather than fully redistributing weight to the remaining modalities. We hypothesize this generalizes to other text-dominant multimodal sentiment/emotion datasets, but we did not run a second-domain experiment to confirm this, and treat it as a hypothesis rather than a claim.

---

## 7. Future work

Distinct from any unfinished current scope:

1. An auxiliary loss penalizing gate weight assigned to masked-out modalities.
2. Architecturally excluding masked features from the gate's softmax rather than relying on soft mask-conditioning.
3. Curriculum-style missingness scheduling during training (e.g., ramping training-time missingness rate up over epochs, rather than uniform sampling from epoch 1).
4. Validation on a second dataset (e.g., CMU-MOSEI or IEMOCAP) to test the generalization hypothesis in Section 6.
5. Increasing seed count to 5+ for tighter significance estimates.

---

## References

*[To be completed — MMIN, TFR-Net, gated multimodal fusion, CMU-MOSI, Self-MM feature release, and the chosen post-2023 comparison method all need full citations before submission.]*
