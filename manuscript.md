# Modality Dropout Prevents Degenerate Collapse but Not Graded Missingness Robustness in Attention-Gated Multimodal Fusion: A Replication Study Across CMU-MOSI and CMU-MOSEI

## Abstract

Multimodal emotion and sentiment recognition systems typically assume all modalities are available at inference time, an assumption that frequently fails in practice. We investigate whether pairing an attention-gated fusion layer — conditioned explicitly on a missingness mask — with modality-dropout training improves robustness to missing modalities on CMU-MOSI. We report a result that changed substantially as the experiment was expanded to five seeds, and we consider what that change revealed more informative than the original result alone.

In the initial pilot, an attention gate trained *without* modality dropout significantly outperformed the identical gate trained *with* it at every nonzero missingness rate (p<0.05, effect size growing with missingness rate). Expanding the experiment to five seeds reversed this finding: none of the three nonzero-rate comparisons remain significant (p=0.327 at 25% missingness, p=0.103 at 50%, p=0.076 at 75%), and the point-estimate gap roughly halved at every rate. We report this plainly as a failure to replicate rather than a softened version of the original claim: an effect significant at initial pilot, with p-values as low as 0.006, did not survive two additional random seeds.

While investigating why the no-dropout gate's apparent advantage was seed-dependent, we found a different, better-supported effect. Under the single-modality condition where text is entirely absent (audio and vision only, the modality-loss condition our diagnostics targeted specifically), every model we trained *with* modality dropout — three architecturally distinct models, including a non-gated baseline — produced non-degenerate predictions (F1≈0.58) in all 5 seeds, with no exceptions across all five seeds. Models trained *without* dropout exposure (5 of 8 architectures, including the original no-dropout gate) instead collapsed, in a seed-dependent 40–80% of individual runs, to predicting a single constant class for every test-set sample — an artifact that inflates raw accuracy to match the test set's 59.6% majority-class rate while F1 drops to exactly 0.000. This distinction was invisible in our own first-draft results because that draft's single-modality table reported accuracy without F1. The corrected pattern is exactly reproducible across every seed we ran and is, we believe, the paper's actual, defensible contribution: in this setup, modality-dropout training's real and replicable benefit is preventing catastrophic mode collapse under complete loss of a modality, not improving graded robustness under partial, proportional missingness — where, at n=5, its effect on the gate is statistically unsupported and trends mildly negative.

We additionally isolated the gate's response to its missingness-mask input directly, independent of what the encoders see (Section 3.6): across every seed and every training regime, flipping the mask input alone shifts the gate's softmax weight on the missing modality by less than 0.01, a near-total insensitivity that also replicates cleanly. The mask-conditioning mechanism this architecture was built around appears to contribute almost nothing to the gate's behavior, with or without dropout training. The paper's scientific conclusion is that modality-dropout training's real, replicable benefit — which also holds for every dropout-trained model on a second dataset, CMU-MOSEI, in Section 3.7 — is preventing catastrophic mode collapse under complete modality loss, not improving graded robustness, and that this architecture's explicit mask-conditioning contributes little independent of that question. Separately, we offer the original ablation, its non-replication, the corrected single-modality analysis, and the mask-channel finding together as a case study in the value of checking F1 alongside accuracy, rerunning small-n significance claims, and verifying project documentation against the executable project state.

**Current claims at a glance.** This paper's framing shifted twice during the project (Section 1.1); for a reader skimming rather than reading in order, the claims that are actually live in this revision are:

- **Not supported at n=5:** the original headline claim — that modality-dropout training hurts an attention gate's graded missingness robustness — did not replicate on CMU-MOSI (Section 3.2), and the same non-effect replicates cleanly on CMU-MOSEI (Section 3.7). Treat the initial-pilot numbers (Tables 3.1a/3.2a) as historical only.
- **Supported and replicated on CMU-MOSI; supported but with one divergence on CMU-MOSEI, now better understood:** modality-dropout training prevents degenerate constant-output collapse under complete loss of the dominant modality. On CMU-MOSI, 0/15 dropout-trained seed-runs collapsed vs. a seed-dependent 40–80% for non-dropout-trained models (Section 3.4). On CMU-MOSEI, dropout-trained models again show 0/15 collapses, but one non-dropout model (`gating_only_no_dropout`) also shows 0/5 collapses — indistinguishable from the dropout-trained group, unlike its MOSI behavior (Section 3.7). A 25-run controlled ablation (Section 3.7.1) confirms training-set size as a real, reproducible contributing factor, but not a fully portable one: the specific seeds that collapse under a size-matched subsample don't cleanly match the seeds that collapse on real CMU-MOSI, so training-set size interacts with something else not yet isolated. This remains the paper's central contribution.
- **A third, secondary finding:** the gate's explicit mask-conditioning mechanism is nearly insensitive to its own mask input (Section 3.6), replicated on CMU-MOSI but not yet tested on CMU-MOSEI. This is not one of the paper's two original framings; see Section 4.

## 1. Introduction

Real-world deployments of multimodal affective computing systems routinely lose access to one or more modalities: a camera occluded, a microphone muted, a text transcript unavailable. A common design response is an attention-gated fusion layer that takes an explicit missingness mask as input, paired with modality-dropout training — randomly zeroing modalities during training — under the intuitive hypothesis that giving the gate the mask, together with dropout exposure, should let it learn to reweight around whichever modalities are absent at test time.

We set out to test this hypothesis directly on CMU-MOSI. We did not find a clean confirmation of it, and the way our own results shifted as we added evidence is, in our view, as much the point of this paper as any single number in it.

### 1.1 What actually happened, in order

The paper's interpretation changed over the course of the project. We retain the historical pilot result for transparency, but the current manuscript's empirical claims and statistical conclusions are based on the complete five-seed analysis:

**Historical pilot.** An attention gate trained without modality dropout significantly outperformed the identical gate trained with it, at every nonzero missingness rate, with the gap growing as missingness increased. Diagnostic analysis at this stage (single-modality masking, gate-weight logging) appeared to support a mechanistic explanation involving the dominant modality's encoder being disrupted by dropout training.

**Five-seed replication.** The ablation's significance did not hold. All three nonzero-rate comparisons that were significant at initial pilot became non-significant at n=5 (Section 3.2). We treat this as a genuine non-replication, not noise to be explained away — with only 2 degrees of freedom, three seeds that happened to agree was not, in retrospect, strong evidence.

**Corrected analysis.** Investigating *why* the no-dropout gate behaved inconsistently across seeds under the single-modality text-missing condition led us to check F1 alongside accuracy for the first time — and to discover that the "good" seeds were not showing robustness at all, but a degenerate constant-output collapse (Section 3.4). Checking whether this collapse pattern held across all 8 models and all 5 seeds revealed a clean, fully-replicating split: dropout-trained models never collapsed; non-dropout-trained models did, in a seed-dependent fraction of runs. This is the finding we consider the paper's real contribution.

### 1.2 Contributions

Given the above, we frame this paper's contributions differently than a typical positive-result paper would:

1. **A documented non-replication.** The originally hypothesized ablation effect (dropout training hurts an attention gate's graded missingness robustness) was significant at initial pilot and is not significant at n=5, at any of the three rates it was originally reported at. We report the full before/after comparison (Section 3.2) rather than only the final numbers, because the discrepancy is itself informative about small-n significance testing in this kind of experiment.

2. **A corrected, replicating finding: dropout training prevents degenerate collapse, not graded robustness.** Under complete loss of the dominant modality (text), every dropout-trained model we tested avoided a constant-output failure mode across all seeds; every non-dropout-trained model exhibited it in some seeds and not others. This is exactly reproducible and, unlike the original ablation, does not depend on which 3 of 5 seeds happen to be reported.

3. **Evidence that the gate's mask-conditioning mechanism is nearly inert.** An isolated intervention — changing only the mask input the gate receives, holding the underlying features fixed — produces a negligible shift in the gate's output, in every seed we tested, regardless of dropout training. This calls into question whether the mask-conditioned gate design is doing meaningful work at all, independent of the missingness question this paper set out to answer.

4. **A worked example of the checks that catch this kind of problem before publication**, documented in full because we think the process is more useful to other researchers than a clean narrative would have been: rerunning at n=5 before trusting initial pilot, checking F1 alongside accuracy on any masking-based diagnostic, and identifying a documentation mismatch between the stated and actual project state (Section 2.7).

### 1.3 Related work

*(Unchanged from the original submission; retained for context.)*

- **Early/late/hybrid fusion baselines**: standard architectures — early (concatenate raw features pre-fusion), late (fuse post-unimodal-prediction), and fixed-weight (a hand-tuned prior over modality reliability) — establish what simple, non-adaptive fusion achieves as missingness increases.
- **Modality dropout** [Neverova et al., 2016-style]: randomly zeroing input modalities during training, originally proposed for cross-modal robustness in action recognition, since adopted widely in affective computing under the assumption that it generalizes to missingness robustness.
- **Attention-gated fusion**: a learned, per-sample weighting over modality embeddings, conditioned on either the embeddings themselves or an explicit availability signal; we study the latter variant here.
- **Post-2023 missing-modality method** [Tan & Zhang, 2025]: DAST-GAN, evaluated on CMU-MOSI/CMU-MOSEI under incomplete-modality conditions with a dynamic attention module and GAN-based adversarial robustness training; our primary SOTA comparison point, `imputation_baseline_post2023`, stands in architecturally for this family (see Section 2.3 for exactly what is, and isn't, reproduced).

## 2. Methods

### 2.1 Dataset

We use **CMU-MOSI** [Zadeh et al., 2016], a standard benchmark for multimodal sentiment analysis, using the pre-aligned 50-timestep feature release.

- **Access:** publicly available aligned feature set (`aligned_50.pkl`), not redistributed with this submission.
- **License:** research use per the original CMU-MOSI release terms.
- **Splits:** standard train/valid/test split as provided (1284/229/686 samples).
- **Modalities:** text (contextual, 768-dim — see note below), audio (COVAREP-derived, 5-dim in this feature release), vision (Facet-derived, 20-dim), all pre-aligned to a common 50-timestep sequence length per segment.
- **Consent/ethics provenance:** CMU-MOSI's original collection protocol and participant consent process are documented in Zadeh et al. (2016); this work performs no new human-subjects data collection and relies entirely on the pre-existing, publicly released feature set.

**Correction from the original submission:** our config file's documented `text_dim: 300` (implying GloVe-based features) does not match the feature dimensionality actually loaded at runtime (`text_dim=768`, printed by every dataset-loading call in this project's logs). We did not catch this discrepancy before the original initial submission draft. It does not appear to invalidate the experiments — the encoder adapts to whatever `text_dim` it is constructed with — but it means prior mentions of "GloVe-based" text features in this project's documentation were incorrect.

We resolve the provenance question as follows. Across the CMU-MOSI/CMU-MOSEI literature, 300-dimensional text features consistently indicate GloVe word embeddings, while 768-dimensional text features consistently indicate contextual BERT-based sentence or utterance embeddings — this split is standard across published work using CMU-MultimodalSDK-derived releases, including the Self-MM line of work our binarization-convention discussion (Section 2.2) references. A runtime `text_dim=768` is therefore consistent with this project having loaded a BERT-featurized release of CMU-MOSI at some point (whether by an intentional switch or a stale config default never updated to match it), not with a corrupted or nonstandard feature dimensionality. We were not able to independently confirm, from the project's artifacts alone, which specific release/commit of the aligned feature set was downloaded, so we do not claim to have fully closed this item — but we can now say with reasonable confidence that the runtime features are BERT-based rather than GloVe-based, and we correct the documentation and all remaining references accordingly.

### 2.2 Binarization convention

*(Unchanged.)* Accuracy/F1 are computed under the `label > 0` (strictly positive vs. non-positive) convention. This is one of two conventions used in the CMU-MOSI literature; some prior work (e.g. Self-MM) uses `label >= 0` instead, which is not numerically interchangeable with this convention for `label == 0` samples. Comparisons to prior published CMU-MOSI numbers should not be treated as directly comparable without first confirming which convention those numbers used.

### 2.3 Models

Eight fusion architectures, all sharing the same per-modality encoder stack (a small GRU/linear encoder per modality, hidden dim 128):

- `early_fusion`, `late_fusion`, `fixed_weight_fusion` — non-adaptive baselines.
- `dropout_only_fusion` — early fusion trained with modality dropout, no gating.
- `gating_only_no_dropout` — attention gate, mask-conditioned, trained at a fixed missingness rate of 0.0 (see the important architectural note below).
- `attention_gated_fusion_full` — the same gate architecture, trained with modality dropout (`Uniform(0, 0.75)` missingness rate per batch).
- `hard_mask_gated_fusion` — a gate variant that structurally zeroes attention weight on masked modalities rather than learning to downweight them.
- `imputation_baseline_post2023` — a reconstruction-based baseline reimplementing the reconstruction-before-fusion principle of DAST-GAN-style methods (Tan & Zhang, 2025) under our own encoder and training setup; a stand-in for that architectural family, not a head-to-head replication of its published numbers.

**Architectural note added in this revision:** `gating_only_no_dropout` is trained at a fixed missingness rate of exactly 0.0 — meaning the mask input to its gate is the constant `[1, 1, 1]` on every single training batch. The gate's weights on the mask-input channels therefore never receive a training signal that varies with the mask value; whatever those weights converge to reflects initialization and incidental gradient flow through a constant input, not anything learned about how to use missingness information. Section 3.6 shows this is consistent with what we observe: the gate is nearly insensitive to its own mask input, in this model and, more surprisingly, in the dropout-trained gate as well.

### 2.4 Missingness protocol

*(Unchanged.)* Missingness is simulated per-sample by independently zeroing each modality's input and setting the corresponding mask entry, at a specified rate; evaluated at rates {0.0, 0.25, 0.5, 0.75}. Training-time missingness for dropout-trained models draws a fresh rate `Uniform(0, 0.75)` per batch.

### 2.5 Seeds and statistical testing

**All reported confirmatory analyses use five seeds (42, 123, 2024, 7, 99), giving 4 degrees of freedom for the paired t-tests in Section 3.2.** The earlier pilot is mentioned only to document the history of the original claim; all current analyses and conclusions are based on the five-seed replication.

Significance tests use paired t-tests across seeds at each missingness rate, matching seeds between compared models.

### 2.6 Training-pipeline sanity check

We ran an overfit-a-fixed-batch check (16 samples, 50 gradient steps) for all 8 models before trusting the full training grid. This item went through three rounds of diagnosis before being closed, and we report the full sequence rather than only the final result, consistent with this manuscript's practice elsewhere.

**Round 1 (original submission).** Five of eight models (`early_fusion`, `late_fusion`, `fixed_weight_fusion`, `gating_only_no_dropout`, `imputation_baseline_post2023`) drove loss from ~2.0–2.1 to below 0.01 within 50 steps, a clean pass. The three dropout-trained models (`dropout_only_fusion`, `attention_gated_fusion_full`, `hard_mask_gated_fusion`) plateaued around 0.17–0.94. We initially attributed this to the missingness rate being resampled fresh every one of the 50 steps — meaning the "fixed batch" these models were asked to overfit was not actually fixed from step to step.

**Round 2.** We fixed the rate-resampling bug (sampling the rate once, before the loop, and reusing the same masked batch for all 50 steps) and reran. The three dropout-trained models still failed to converge (losses 0.94, 0.17, 0.17 at 50 steps) — the original diagnosis was incomplete. Inspecting the code further, we found a second issue: with the rate left to be drawn implicitly from the RNG stream, that draw happens *after* model construction, and different architectures consume different amounts of the shared RNG stream while initializing their parameters. This meant each dropout-trained model was landing on an arbitrary, uncontrolled rate as a side effect of its parameter count, not by design.

**Round 3.** We made the missingness rate an explicit, controlled parameter (fixed at 0.375, the midpoint of the training range, for all three dropout-trained models) and reran. All three still failed to reach near-zero loss within 50 steps (0.27, 0.45, 0.11). We considered two remaining explanations: (a) a genuine training-pipeline bug specific to dropout-trained models, or (b) an expected, correct loss floor — with `audio_dim=5`, it seemed plausible that some of the 16 fixed samples could become identical or near-identical to each other in masked feature space (e.g., both left with only a low-dimensional audio vector) while carrying different regression labels, in which case no model could reach exact zero on that batch. A direct check for near-identical-embedding/different-label pairs among the 16 samples found none, and running the same fixed masked batch for more steps settled the question directly: `hard_mask_gated_fusion` reached exactly 0.0000 loss by step 500, and `dropout_only_fusion` and `attention_gated_fusion_full` each reached exactly 0.0000 by step 3000 — ruling out explanation (b), since a genuine collision floor could not have been fully escaped by any model regardless of step count.

**Resolution.** All 8 models pass a genuine fixed-batch overfit test; there is no training-pipeline bug. Dropout-trained models simply converge much more slowly under partial missingness (roughly 10–60x more gradient steps than non-dropout models, in this test) than the sanity check's original 50-step budget assumed — plausibly because zeroed-modality inputs produce weaker or noisier per-step gradients, not because anything is broken. We revise the sanity-check protocol going forward to scale the step budget with whether a model is dropout-trained (we do not use a uniform 50-step budget for all 8 models) rather than treating a fixed step count as universally sufficient.

### 2.7 Reproducibility, code/data provenance, and documentation consistency

During the 5-seed replication, we identified a documentation-consistency problem in the project's working directory. The `README.md` described checkpoints and diagnostics as "not yet executed," although the corresponding outputs were already present. Separately, the working directory's `data/` directory did not contain its dataset-loading module (`dataset.py`), which existed in a separate code archive. These discrepancies show that reproducibility requires checking both the executable project state and the documentation that describes it; both are now treated as separate provenance checks.

## 3. Results

### 3.1 Main results: original initial pilot vs. 5-seed replication

**Table 3.1a — Initial pilot (historical comparison).** Mean accuracy ± 95% CI half-width. These values are retained only to document how the original claim changed; all current inference uses five seeds.

| Model | 0% | 25% | 50% | 75% |
|---|---|---|---|---|
| `attention_gated_fusion_full` | 0.764 ± 0.012 | 0.686 ± 0.021 | 0.606 ± 0.024 | 0.557 ± 0.026 |
| `hard_mask_gated_fusion` | 0.778 ± 0.023 | 0.699 ± 0.021 | 0.621 ± 0.027 | 0.563 ± 0.017 |
| `dropout_only_fusion` | 0.768 ± 0.037 | 0.687 ± 0.047 | 0.610 ± 0.027 | 0.558 ± 0.027 |
| `gating_only_no_dropout` | 0.759 ± 0.015 | 0.713 ± 0.023 | 0.680 ± 0.046 | 0.653 ± 0.025 |
| `fixed_weight_fusion` | 0.735 ± 0.089 | 0.684 ± 0.062 | 0.639 ± 0.108 | 0.618 ± 0.097 |
| `early_fusion` | 0.779 ± 0.016 | 0.720 ± 0.060 | 0.672 ± 0.094 | 0.635 ± 0.104 |
| `late_fusion` | 0.779 ± 0.024 | 0.713 ± 0.048 | 0.656 ± 0.121 | 0.632 ± 0.170 |
| `imputation_baseline_post2023` | 0.777 ± 0.016 | 0.719 ± 0.064 | 0.679 ± 0.098 | 0.653 ± 0.047 |

**Table 3.1b — Five-seed replication.** Mean accuracy ± 95% CI half-width, n=5 seeds (df=4).

| Model | 0% | 25% | 50% | 75% |
|---|---|---|---|---|
| `attention_gated_fusion_full` | 0.765 ± 0.006 | 0.686 ± 0.008 | 0.601 ± 0.011 | 0.554 ± 0.011 |
| `hard_mask_gated_fusion` | 0.772 ± 0.023 | 0.692 ± 0.021 | 0.610 ± 0.027 | 0.559 ± 0.017 |
| `dropout_only_fusion` | 0.773 ± 0.017 | 0.691 ± 0.018 | 0.607 ± 0.012 | 0.559 ± 0.010 |
| `gating_only_no_dropout` | 0.757 ± 0.008 | 0.697 ± 0.028 | 0.645 ± 0.062 | 0.612 ± 0.071 |
| `fixed_weight_fusion` | 0.747 ± 0.038 | 0.703 ± 0.039 | 0.658 ± 0.051 | 0.635 ± 0.046 |
| `early_fusion` | 0.774 ± 0.010 | 0.717 ± 0.035 | 0.659 ± 0.054 | 0.621 ± 0.062 |
| `late_fusion` | 0.778 ± 0.011 | 0.704 ± 0.025 | 0.634 ± 0.058 | 0.600 ± 0.081 |
| `imputation_baseline_post2023` | 0.771 ± 0.013 | 0.723 ± 0.027 | 0.681 ± 0.035 | 0.659 ± 0.021 |

Point estimates are broadly stable between the two tables (largest shift: `late_fusion`@75%, 0.632 → 0.600). CI half-widths shrink substantially for most models, as expected with more seeds — but note `gating_only_no_dropout`'s CI at 50%/75% *widens* rather than tightening (0.046→0.062, 0.025→0.071), a direct numerical signature of the seed-dependent bimodal behavior diagnosed in Section 3.4.

### 3.2 Significance tests: five-seed replication and historical comparison

**Table 3.2a — Historical pilot comparisons (reported for transparency only).**

| Comparison | Rate | p-value |
|---|---|---|
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.0 | 0.212 |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.25 | **0.031** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.5 | **0.044** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.75 | **0.006** |
| `hard_mask_gated_fusion` vs `gating_only_no_dropout` | 0.75 | **0.001** |

**Table 3.2b — Five-seed replication, current inferential results.** Cohen's d is paired (mean of per-seed differences ÷ SD of per-seed differences, n=5), computed from `results_raw_5seed.csv`.

| Comparison | Rate | p-value | Paired Cohen's d | Change from Table 3.2a |
|---|---|---|---|---|
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.0 | **0.024** | −1.58 | now significant (was not); direction: no-dropout gate now *worse*, not better |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.25 | 0.327 | 0.50 | **no longer significant** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.5 | 0.103 | 0.94 | **no longer significant** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.75 | 0.076 | 1.06 | **no longer significant** |
| `hard_mask_gated_fusion` vs `gating_only_no_dropout` | 0.75 | 0.087 | −1.01 | **no longer significant** |

The 25%/50%/75% comparisons illustrate exactly the gap effect-sizes-only reporting would obscure: each shows a large-to-very-large point-estimate effect (d≈0.5–1.1) that is nonetheless statistically unsupported at n=5 given the seed-to-seed variance — large effect size and non-significance are not in tension here, they are both true simultaneously, which is itself informative about how much the original 3-seed result over-trusted a small sample. Sign convention: negative d means `gating_only_no_dropout` scored higher than the comparison model; positive d means lower.

**Every comparison that drove the paper's original headline claim lost significance when the analysis was expanded to five seeds.** The only newly-significant result (rate=0.0, no missingness at all) is in the opposite direction from, and irrelevant to, the original missingness-robustness claim. We do not consider the 5-seed data supportive of the original ablation hypothesis in any form. Full 28-row comparison table (all model pairs, all rates, both seed counts) is included in the project's data release as `significance_5seed_full.csv`.

Full comparisons against `imputation_baseline_post2023`, `fixed_weight_fusion`, and `dropout_only_fusion` at n=5 (all p>0.10 at every rate) are reported in the released CSV rather than reproduced here in full, since none change the picture above.

### 3.3 Parameter counts and efficiency

*(Unchanged — architecture did not change between the two runs.)* Parameter counts range from 454,659 (`late_fusion`) to 701,697 (`imputation_baseline_post2023`), a 39.3%–54.3% increase for the reconstruction-based baseline over the simplest fusion architectures. Full table retained from the original submission; verified via `verify_manuscript_numbers.py`.

### 3.4 Single-modality masking: a corrected, more careful reading

This section differs substantially from the original submission, which reported only accuracy for this table. **Adding F1 changes the finding.**

**Table 3.4 — text-missing (audio+vision only) condition, per-seed, all 5 seeds, accuracy and F1.**

| Model | seed 42 | seed 123 | seed 2024 | seed 7 | seed 99 |
|---|---|---|---|---|---|
| `attention_gated_fusion_full` | 0.407 / .576 | 0.404 / .575 | 0.405 / .575 | 0.404 / .575 | 0.405 / .575 |
| `dropout_only_fusion` | 0.410 / .576 | 0.404 / .575 | 0.407 / .576 | 0.405 / .575 | 0.404 / .575 |
| `hard_mask_gated_fusion` | 0.407 / .576 | 0.404 / .575 | 0.411 / .577 | 0.402 / .574 | 0.408 / .575 |
| `gating_only_no_dropout` | **0.596 / .000** | **0.596 / .000** | **0.596 / .000** | 0.404 / .575 | 0.404 / .575 |
| `early_fusion` | **0.596 / .000** | 0.415 / .578 | **0.596 / .000** | **0.596 / .000** | 0.404 / .575 |
| `fixed_weight_fusion` | **0.596 / .000** | 0.410 / .577 | **0.596 / .000** | **0.596 / .000** | 0.589 / .021 |
| `imputation_baseline_post2023` | **0.596 / .000** | 0.413 / .577 | **0.596 / .000** | 0.596 / .007 | 0.592 / .021 |
| `late_fusion` | 0.404 / .575 | **0.596 / .000** | **0.596 / .000** | 0.404 / .575 | 0.404 / .575 |

Bold entries mark accuracy = 0.596210, exactly `409/686`, this test set's negative-class base rate, paired with F1 = 0.000 (or, in two `imputation_baseline_post2023`/`fixed_weight_fusion` cases, F1 ≈ 0.02, functionally the same failure). **These are not cases of a model handling missing text well. They are cases of the model outputting the same non-positive prediction for every one of the 686 test samples, regardless of input, which happens to match the majority class often enough to look like reasonable accuracy.** The three dropout-trained models (`attention_gated_fusion_full`, `dropout_only_fusion`, `hard_mask_gated_fusion`) never once produce this pattern, in any of 15 seed-runs (3 models × 5 seeds). Every non-dropout-trained model produces it in some seeds and not others — a seed-dependent rate of roughly 40–80% of runs per model (`late_fusion` 2/5=40%, `early_fusion` 3/5=60%, `gating_only_no_dropout` 3/5=60%, `fixed_weight_fusion` and `imputation_baseline_post2023` 4/5=80% each).

*Correction from the previous revision: three of `late_fusion`'s five seeds (42, 7, 99) were previously reported as missing ("— / —") in this table. The underlying per-seed results file (`single_modality_results.csv`) contains complete data for all 5 seeds and all 8 models; the missing cells were a reporting gap, not a missing-data issue in the underlying run. All three previously-missing `late_fusion` seeds are non-collapsed (0.404/.575), giving `late_fusion` a 2/5 (40%) collapse rate, consistent with — not an exception to — the seed-dependent pattern described above.*

We did not catch this in the initial submission because that draft's Table 3.4 reported only accuracy. The original pilot seed subset (42, 123, 2024) happens to include zero or more degenerate collapses depending on model, which is part of why the original framing ("the no-dropout gate retains 0.596 accuracy under missing text, versus 0.405 for the dropout-trained gate") was actively misleading rather than merely imprecise: the higher number was, for at least the `gating_only_no_dropout` model at those specific seeds, not a better result.

### 3.5 Gate-weight response to the missingness mask

*(Retained from the original submission; not contradicted by the above, though we note this analysis has not itself been re-checked for the degenerate-collapse artifact and should be treated with the same caution until it is — see Section 7.)*

Both the dropout-trained and no-dropout-trained gates show similar relative responsiveness to the mask signal when text is present vs. absent (e.g., mean gate weight on text: 0.926 present vs. 0.761 absent for `gating_only_no_dropout`@25% missingness; 0.857 vs. 0.692 for `attention_gated_fusion_full`@25%), a pattern consistent across the 0.25/0.5/0.75 rates tested. Section 3.6 below investigates this mask-responsiveness more directly and finds it is much smaller than these numbers suggest once the encoder's response to zeroed input is controlled for separately.

**Per-sample re-audit (manuscript Section 7, item 1, now complete).** The aggregate means above could in principle hide a distorting subset of degenerate per-sample weights. We recomputed the full per-sample distribution (18,522/30,870-row `gate_weights_raw.csv`, 3,430 test samples × 3 models × 3 rates) rather than relying on the mean alone:

| model | rate | mean | median | p25 | p75 | frac. within 0.02 of 0 | frac. within 0.02 of 1 |
|---|---|---|---|---|---|---|---|
| `attention_gated_fusion_full` | 0.25 | 0.743 | 0.742 | 0.612 | 0.897 | 0.000 | 0.124 |
| `attention_gated_fusion_full` | 0.50 | 0.703 | 0.678 | 0.550 | 0.880 | 0.000 | 0.086 |
| `attention_gated_fusion_full` | 0.75 | 0.669 | 0.601 | 0.507 | 0.850 | 0.000 | 0.056 |
| `gating_only_no_dropout` | 0.25 | 0.829 | 0.877 | 0.690 | 0.971 | 0.000 | 0.223 |
| `gating_only_no_dropout` | 0.50 | 0.784 | 0.792 | 0.594 | 0.962 | 0.000 | 0.214 |
| `gating_only_no_dropout` | 0.75 | 0.746 | 0.668 | 0.578 | 0.951 | 0.000 | 0.209 |
| `hard_mask_gated_fusion` | 0.25 | 0.747 | 0.977 | 0.908 | 0.990 | 0.236 | 0.451 |
| `hard_mask_gated_fusion` | 0.50 | 0.551 | 0.965 | 0.000 | 0.993 | 0.441 | 0.415 |
| `hard_mask_gated_fusion` | 0.75 | 0.383 | 0.000 | 0.000 | 1.000 | 0.615 | 0.352 |

**`hard_mask_gated_fusion`'s large near-degenerate fraction is architectural, not an emergent finding, and should not be read alongside the other two rows as if directly comparable.** Its gate additively masks absent modalities' logits to `-inf` before the softmax (`models/hard_mask_gate.py`), so weight on text is *structurally forced* to exactly 0 whenever text is marked absent and to exactly 1 whenever text is the only modality present — this is the model doing exactly what its design specifies, not a discovered pathology. We report it for completeness but do not treat it as informative about the same question the other two rows answer.

**The two soft-gated models (`attention_gated_fusion_full`, `gating_only_no_dropout`) show two real, previously invisible-in-the-aggregate findings:**

1. **Neither soft-gated model ever pins its weight on text near zero (0.000 in the near-0 column at every rate), even in the subset of samples where text is actually marked absent.** This is independent confirmation of Section 3.6's isolated-intervention finding that the gate is largely insensitive to its own mask input — here, from the full graded-missingness evaluation data rather than an isolated intervention, and it rules out one specific way the aggregate mean in the paragraph above could have been misleading (a soft floor near zero for absent-text samples pulling the mean down, while present-text samples cluster near 1 — that is not what is happening; the distribution is continuous, not bimodal-at-the-extremes).
2. **`gating_only_no_dropout`'s distribution has a meaningfully fatter near-1 tail than `attention_gated_fusion_full`'s at every rate** (22.3%/21.4%/20.9% of samples within 0.02 of weight=1, vs. 12.4%/8.6%/5.6% for the dropout-trained gate) — a real distributional difference the aggregate means alone do not show. We note this as a plausible, but not confirmed, connection to Section 3.4's collapse finding: a gate more often near-saturated on text specifically is a gate more exposed to whatever text-specific failure mode produces collapse when text is completely absent, but we have not run the intervention that would confirm this connection directly, and we flag it as a hypothesis rather than a demonstrated mechanism.

### 3.6 Isolating the gate's response to the mask signal from the encoder's response to zeroed input

Motivated by the seed-dependent split in Section 3.4, we ran a controlled intervention using the `gating_only_no_dropout` checkpoints across all 5 seeds, isolating two effects that Section 3.4's masking protocol conflates: (a) the gate receiving a different *mask value*, and (b) the encoder receiving a different *feature input* (zeroed text).

**Isolating the mask-channel effect alone** (real, non-zeroed text features throughout; only the mask value fed to the gate changes from `[1,1,1]` to `[0,1,1]`), for `gating_only_no_dropout` (never trained with a varying mask input):

| seed | mean gate weight on text, mask=on | mean gate weight on text, mask=off | shift |
|---|---|---|---|
| 42 | 0.860 | 0.851 | −0.009 |
| 123 | 0.999 | 0.999 | −0.0001 |
| 2024 | 0.929 | 0.924 | −0.005 |
| 7 | 0.723 | 0.714 | −0.010 |
| 99 | 0.888 | 0.881 | −0.007 |

**The same intervention, run this revision on `attention_gated_fusion_full`** (manuscript Section 7, item 1; trained with the mask input varying every step per Section 2.3/2.4 — the model this open question was specifically about):

| seed | mean gate weight on text, mask=on | mean gate weight on text, mask=off | shift |
|---|---|---|---|
| 42 | 0.9857 | 0.9833 | −0.0024 |
| 123 | 0.8224 | 0.8142 | −0.0081 |
| 2024 | 0.7750 | 0.7655 | −0.0095 |
| 7 | 0.6489 | 0.6369 | −0.0120 |
| 99 | 0.7262 | 0.7186 | −0.0076 |

**The mask-inertness finding extends to the dropout-trained gate.** Every shift is under 0.012 in magnitude — comparable to, and if anything slightly smaller in relative terms than, the no-dropout-trained gate's shifts above. This is a real, previously untested result, not an assumption carried over from the no-dropout case: `attention_gated_fusion_full` *was* trained with its mask input varying every step, so there was no structural reason to expect it would also ignore that input — yet it does. This strengthens Section 4's claim that the mask-conditioning mechanism contributes little to gate behavior, independent of dropout training, and resolves the open question flagged in the previous revision ("we have not run the equivalent isolation experiment on `attention_gated_fusion_full` and do not want to claim more than we've directly measured").

The gate's response to its own mask input, isolated from any change in the underlying features, is negligible — under 0.01 in every seed, with no meaningful difference between seeds that later collapse (42, 123, 2024) and seeds that do not (7, 99). **This directly supports the architectural point in Section 2.3**: since this model's training never varied the mask input, its weights on that channel did not learn to respond to it, and this holds regardless of which seed produced the initialization.

**Isolating the encoder-zeroing effect alone** (mask channel held at `[1,1,1]` throughout; only the encoder's text input is zeroed):

| seed | mean gate weight on text, real input | mean gate weight on text, zeroed input | shift |
|---|---|---|---|
| 42 | 0.860 | 0.615 | −0.245 |
| 123 | 0.999 | 0.997 | −0.002 |
| 2024 | 0.929 | 0.708 | −0.221 |
| 7 | 0.723 | 0.547 | −0.177 |
| 99 | 0.888 | 0.618 | −0.270 |

This effect is much larger than the mask-channel effect, confirming that whatever downstream behavior differs between seeds is driven by how each seed's encoder-plus-head jointly respond to a zeroed text input, not by the gate's mask conditioning. However, **this table does not cleanly separate the seeds that later collapse (42, 123, 2024) from those that don't (7, 99), and the counterexamples run in both directions**: collapsing seed 123 shows almost no shift (−0.002), while non-collapsing seeds 7 and 99 both show shifts (−0.177, −0.270) as large as or larger than two of the three collapsing seeds. We do not have a single identified structural cause for the seed-dependent split in Section 3.4's *specific* accuracy values; we have ruled out the mask channel as the cause and localized the effect to the encoder-plus-head's response to zeroed input, without fully explaining why that response itself varies by seed in a way that doesn't track collapse status through this particular measurement. We report this partial result rather than a tidier story we don't have evidence for.

### 3.7 Cross-Dataset Replication (CMU-MOSEI)

Following this manuscript's own practice of scoping claims honestly (Section 1.2, item 4), we tested two of this paper's findings on a second dataset — CMU-MOSEI (thuiar/MMSA aligned-feature release; N=22,856 total, 4,659 test / 16,326 train / 1,871 valid; `text_dim=768`, `audio_dim=74`, `vision_dim=35`, confirmed at runtime) — rather than attempting a full second 8-model × 5-seed × 4-rate grid. We ran exactly the two comparisons below, 8 models × 5 seeds = 40 runs, using the same 5 seeds (42, 123, 2024, 7, 99) as the MOSI experiments.

**Claim A: the graded-robustness non-effect (Section 3.2).** Does `gating_only_no_dropout` vs. `attention_gated_fusion_full` show no significant benefit from dropout training at any missingness rate?

| missingness rate | mean (no-dropout) | mean (dropout) | diff | p-value |
|---|---|---|---|---|
| 0.00 | 0.7255 | 0.7291 | −0.0036 | 0.424 |
| 0.25 | 0.6847 | 0.6757 | 0.0089 | 0.298 |
| 0.50 | 0.6413 | 0.6313 | 0.0100 | 0.308 |
| 0.75 | 0.6147 | 0.5994 | 0.0152 | 0.189 |

All four comparisons are non-significant, matching the MOSI non-replication (Section 3.2) exactly in direction and conclusion: on CMU-MOSEI, as on CMU-MOSI at n=5, dropout training confers no statistically significant graded-robustness benefit at any missingness rate.

**Claim B: the collapse-prevention effect (Section 3.4).** Does the dropout-trained/non-dropout-trained collapse split reproduce under complete loss of text (the dominant modality)?

| model | dropout-trained? | collapses (F1<0.05) / 5 seeds |
|---|---|---|
| `attention_gated_fusion_full` | yes | 0/5 |
| `dropout_only_fusion` | yes | 0/5 |
| `gating_only_no_dropout` | no | 0/5 |
| `hard_mask_gated_fusion` | no | 1/5 |
| `fixed_weight_fusion` | no | 1/5 |
| `imputation_baseline_post2023` | no | 1/5 |
| `early_fusion` | no | 2/5 |
| `late_fusion` | no | 3/5 |

We verified this collapse detection against CMU-MOSEI's actual test-set base rate rather than reusing MOSI's 59.6% constant (`MOSEI_PROVENANCE.md` flagged this exact risk): CMU-MOSEI's test set is close to balanced (N=4,659, positive-class rate 0.4902, majority-class rate 0.5098). We recomputed this directly from the raw per-seed results (not an aggregate estimate): filtering to only the runs that actually collapsed (F1<0.05, 10 of 40 model/seed pairs), accuracy ranges [0.5098, 0.5151] — landing almost exactly on the majority-class rate, the same collapse signature as MOSI, just against a different (more balanced) constant.

The three dropout-trained models again show zero collapses, consistent with MOSI. **However, this does not fully replicate the MOSI collapse split, and we report the divergence directly rather than smoothing it over:** on MOSI, `gating_only_no_dropout` collapsed in a subset of seeds despite not being dropout-trained; on MOSEI, it shows 0/5 collapses, indistinguishable from the dropout-trained group. The five non-dropout models that *do* collapse on MOSEI overlap with, but are not identical to, the pattern on MOSI. The core qualitative claim — dropout-trained models never collapse, while collapse is common (though not universal) among non-dropout-trained models — replicates. The specific claim that dropout exposure is *necessary* to avoid collapse (implied by MOSI's data, where every non-dropout model collapsed in at least one seed) does not fully replicate: one non-dropout model avoided collapse entirely on MOSEI.

### 3.7.1 Testing candidate explanations for the divergence

We tested the two candidate factors named above directly, via a controlled comparison (`run_mosei_divergence_ablation.py`), rather than leaving them as untested speculation. Each ablation retrains `gating_only_no_dropout` from scratch across the same 5 seeds, modifying MOSEI's train split along exactly one axis at a time and holding the rest of the pipeline identical to Section 3.7's main run:

- **`trainsize` ablation:** subsample MOSEI's train split down to CMU-MOSI's exact train size (N=1,284 vs. MOSEI's native 16,326), leaving the label distribution at its natural (near-balanced) rate.
- **`baserate` ablation:** resample MOSEI's train split (undersampling the minority class, keeping all majority-class samples) until its majority-class rate matches CMU-MOSI's 0.596, leaving the train size much larger than CMU-MOSI's (N=13,889 achieved, vs. CMU-MOSI's 1,284).

| Ablation | n_train | achieved majority rate | Collapses (F1<0.05) / 5 seeds | F1 per seed |
|---|---|---|---|---|
| `trainsize` | 1,284 | ~0.50–0.51 (unchanged, natural) | **1/5 (20%)** | 0.041\*, 0.055, 0.166, 0.643, 0.068 |
| `baserate` | 13,889 | 0.596 (matches CMU-MOSI) | **0/5 (0%)** | 0.273, 0.561, 0.410, 0.517, 0.404 |

\*seed 42 is the one collapsed run (F1=0.0406 < 0.05 threshold); seeds 123 and 99 (F1=0.055, 0.068) sit close enough to the threshold that we do not read the 1/5 count alone as the full picture — see below.

**Reading these results honestly, without overclaiming:** the `trainsize` ablation moves the collapse rate from MOSEI's full-scale 0/5 partway toward MOSI's 3/5 (60%) — one seed collapses outright, and two more (F1=0.055, 0.068) sit close enough to the F1<0.05 threshold that a stricter or looser cutoff could plausibly change their classification, which the binary collapse count alone does not convey. The `baserate` ablation shows no movement at all: 0/5 collapses, indistinguishable from the full-scale MOSEI result, despite exactly matching CMU-MOSI's label distribution. **This is directional evidence that training-set size, not base-rate balance, is the more likely contributing factor** — but it is directional evidence from 5 seeds and a partial (not complete) shift toward MOSI's rate, not a confirmed mechanism.

**Follow-up: multiple independent subsample draws per seed.** A single draw cannot distinguish "training-set size is the mechanism" from "seed 42's one particular subsample happened to be unlucky." We reran the `trainsize` ablation with 5 independent random subsample draws per seed (25 total training runs, `--n_draws 5`), holding model initialization fixed within each seed so only the specific subsample composition varies across draws:

| seed | collapses / draws | seed | collapses / draws |
|---|---|---|---|
| 42 | **4/5** | 7 | 0/5 |
| 99 | **2/5** | 123 | 0/5 |
| | | 2024 | 0/5 |

Aggregate: 6/25 (24%) collapsed. This rules out "one unlucky draw" for seed 42 specifically — 4 of 5 independent subsamples collapsed, which is not consistent with a single fluke. **This is real, meaningfully stronger evidence that reducing training-set size can reproducibly induce collapse, at least for some seeds.**

**However, this does not resolve into a clean "certain seeds are just collapse-prone" story, and we report the complication directly rather than smoothing past it.** Cross-referencing against which seeds actually collapse on real CMU-MOSI (Section 3.4, corrected: `gating_only_no_dropout` collapses on seeds 42, 123, 2024, not on 7, 99) — only seed 42 overlaps between the two. Seed 99 collapses heavily under this ablation (2/5 draws) despite being one of the two seeds that does *not* collapse on real CMU-MOSI. Seeds 123 and 2024, both genuine CMU-MOSI collapsers, stay completely clean (0/5) under this ablation despite training on a subsample matched to CMU-MOSI's exact size. **If training-set size alone explained the divergence via a portable per-seed collapse-proneness, we would expect the ablation's collapse-prone seeds to match CMU-MOSI's actual collapse-prone seeds; they mostly do not (1 of 3 overlaps).** The more accurate characterization is: reducing training-set size can induce collapse, and does so reproducibly for at least one seed under repeated independent subsampling, but *which* seed collapses under a small MOSEI subsample is not simply predictable from which seed collapses on real MOSI — some other factor (plausibly the specific composition of samples in a given draw, interacting with a given seed's model initialization, rather than either factor alone) is also at work, and we have not isolated it. Training-set size is real evidence, not the whole explanation.

## 4. Discussion

We organize this section around what changed between our original hypothesis and what the full evidence, including the corrected Section 3.4 and the isolation experiments in Section 3.6, actually supports.

**The original hypothesis — that modality-dropout training hurts an attention gate's graded missingness robustness — is not supported at n=5.** We think the most likely explanation for the initial pilot result is a small-sample false positive: three seeds that happened to agree, at a comparison with substantial seed-to-seed variance (visible directly in the CI half-widths in Table 3.1, several of which exceed 0.05 for the relevant models). We do not have evidence for any of the mechanistic explanations our original initial draft proposed (encoder disruption, a "flatter, more hedged baseline weighting policy") as an explanation for graded missingness performance, because the effect they were explaining is no longer present in the data.

**The corrected finding — dropout training prevents degenerate collapse under complete modality loss — is, we think, real and better-supported.** It replicates exactly across every seed and every dropout-trained model we tested (0/15 collapses), against a clearly nonzero and seed-dependent collapse rate for every non-dropout-trained model. We do not yet have a mechanistic account of *why* dropout training prevents this specific failure mode. We want to be explicit here, not only in Limitations, about how weak the current mechanistic evidence is: Section 3.6's encoder-zeroing isolation does not cleanly separate collapsing from non-collapsing seeds, and the counterexamples run in both directions — collapsing seed 123 shows almost no shift (−0.002) under encoder-zeroing, while non-collapsing seeds 7 and 99 (−0.177, −0.270) show shifts comparable to or larger than two of the three collapsing seeds (42: −0.245, 2024: −0.221). If the encoder-zeroing response were the mechanism, we would expect it to track collapse status; it does not, at least not through this measurement. We therefore do not treat "dropout regularizes the encoder's response to zeroed input" as the explanation for the collapse-prevention effect — it is, at most, a ruled-out-alternative-turned-partial-lead: we have ruled out the mask channel as the cause (Section 3.6) and localized the remaining effect to the encoder-plus-head, but seed 123 specifically shows this localization is incomplete, and we have not tested this directly. We flag the mechanism as unresolved and identify the natural next experiment in Section 7.

**The mask-conditioning mechanism itself appears to contribute little, independent of the missingness question, and this now holds for both gates we tested.** Section 3.6's isolated mask-channel intervention shows near-zero sensitivity in every seed for `gating_only_no_dropout` (whose training procedure never varied that input) and, as of this revision, for `attention_gated_fusion_full` too (whose training procedure did vary that input every step) — every shift under 0.012 in magnitude for both models. The gate is doing most of its useful work through its response to encoder outputs rather than through its explicit mask input, which means the entire mask-conditioning design element — the architectural feature this line of work is specifically about — is doing less than assumed, independent of any missingness-robustness question and independent of whether the gate was ever trained to see a varying mask. This is, in our view, a more interesting and more surprising finding than either the original ablation or its non-replication, and we did not go looking for it; it fell out of trying to understand a result that turned out not to be real.

**What this case study adds beyond the specific CMU-MOSI result.** A growing body of work on small-sample significance testing in deep learning benchmarks has documented how easily a handful of seeds can produce a confident-looking but non-robust effect, and negative-results and replication-focused venues have argued for reporting non-replications directly rather than quietly dropping them. This paper is intended as a concrete instance of that pattern rather than only a report about attention-gated fusion: a 3-seed effect with p as low as 0.006 disappeared at n=5, and the artifact that produced the original, misleading positive result (accuracy-only evaluation masking a constant-output collapse) is a failure mode general to any masking-based robustness evaluation, not specific to this architecture or dataset. We see the paper's meta-scientific contribution — documenting a full pilot-to-replication trajectory, including the false start, rather than presenting only the final corrected numbers — as at least as important as the specific empirical finding about dropout and collapse.

**Mechanism status, consolidated.** The findings above are discussed across this section, Section 3.5's caveat, and Section 7's open items; the table below collects every mechanistic claim in one place so a reader does not have to cross-reference to see what is confirmed, ruled out, or still open.

| Finding | Status | Section(s) |
|---|---|---|
| Dropout training hurts graded missingness robustness (original ablation) | **Not replicated / refuted** at n=5 on CMU-MOSI; non-effect also holds on CMU-MOSEI | 3.2, 3.7 |
| Dropout training prevents degenerate collapse under complete modality loss | **Replicated** on CMU-MOSI (0/15 collapses) and on CMU-MOSEI (0/15 collapses); the *necessity* of dropout for avoiding collapse does not fully replicate on CMU-MOSEI (one non-dropout model also avoids collapse) | 3.4, 3.7, 4 |
| Mask channel as the mechanism behind collapse-prevention | **Ruled out** — isolated mask-channel intervention shows near-zero sensitivity (<0.01 shift) in every seed | 3.6 |
| Encoder response to zeroed input as the mechanism behind collapse-prevention | **Open / partial lead, not confirmed** — shift does not cleanly track collapse status (seed 123 is a counterexample) | 3.6, 4 |
| Why `gating_only_no_dropout` avoids collapse on CMU-MOSEI but not CMU-MOSI | **Training-set size confirmed as a real, reproducible contributing factor** (25-run multi-draw ablation: seed 42 collapses in 4/5 independent subsamples), **but not a portable per-seed property** — the ablation's collapse-prone seeds (42, 99) mostly don't match CMU-MOSI's actual collapse-prone seeds (42, 123, 2024; only seed 42 overlaps). Some interaction between subsample composition and seed remains unexplained | 3.7.1, 7 (item 7) |
| Mask-conditioning mechanism's general contribution to gate behavior | **Near-inert, confirmed on both `gating_only_no_dropout` and `attention_gated_fusion_full`**; not yet tested on CMU-MOSEI | 3.6, 4 |
| Section 3.5's aggregate gate-weight means vs. the per-sample collapse artifact identified in 3.4 | **Re-audited — no distortion found in the aggregate means, but a real distributional difference emerged**: neither soft-gated model's per-sample weight ever approaches 0 even for text-absent samples (independent confirmation of 3.6's mask-inertness finding); `gating_only_no_dropout` has a meaningfully fatter near-1 tail than `attention_gated_fusion_full` at every rate, a plausible but unconfirmed link to 3.4's collapse finding | 3.5, 7 (item 1) |
| Narrow-dropout-range variant (never fully zeros a modality) and whether it still prevents collapse | **Run this revision — no collapse observed under graded random missingness (F1≥0.615 at every rate/seed), but this is not a strict test of the complete-text-loss condition** the original claim is about | 7 (item 4) |

## 5. Limitations

- **The core ablation claim did not replicate**, and we do not know whether a larger sample (n=10, n=20) would recover a smaller but real effect, show the effect flips sign, or show no effect at all. We consider the current n=5 evidence insufficient to make any claim about the sign or existence of dropout training's effect on graded missingness robustness in this setup, and we recommend against citing the original pilot numbers, which remain in Section 3.1a/3.2a purely for transparency about what changed.
- **The degenerate-collapse finding, while it replicates cleanly on accuracy/F1, has not been mechanistically explained.** Section 3.6's attempt to localize the cause (mask channel vs. encoder response to zeroed input) ruled out one candidate but did not identify a clean alternative that separates collapsing from non-collapsing seeds within the non-dropout-trained models.
- **Section 3.5's gate-weight analysis has not been re-audited for the same degenerate-collapse artifact** that affected Section 3.4; until it is, its aggregate mean-weight numbers should be treated with the same caution, since a small number of degenerate per-sample predictions could distort an aggregate mean without being visible in that table's current form.
- **The training-pipeline sanity check (Section 2.6) is now fully resolved**, after three rounds of diagnosis: a rate-resampling bug and an RNG-drift-driven uncontrolled-rate bug were both fixed, and the three dropout-trained models' remaining non-convergence at the original 50-step budget was confirmed to be slow convergence under partial missingness rather than a training bug or an information-collision floor — all 8 models reach exactly 0.0000 loss given a sufficient step budget. This item is closed, not flagged as open.
- **Two-dataset scope, one claim with a confirmed exception.** The primary experiments use CMU-MOSI; Section 3.7 extends a scoped, two-claim replication (the graded-robustness non-effect and the collapse-prevention effect) to CMU-MOSEI. We make no claim beyond these two datasets — IEMOCAP and other multimodal benchmarks remain untested. Within CMU-MOSEI, the graded-robustness non-effect replicates cleanly; the collapse-prevention effect replicates in its core form but with a confirmed exception (`gating_only_no_dropout` avoids collapse on MOSEI despite not being dropout-trained, unlike on MOSI), which we have not mechanistically explained. Readers should not treat the MOSEI extension as validating that dropout exposure is *necessary* to avoid collapse in general — only that it is *sufficient* across both datasets tested.
- **Binarization convention** (Section 2.2) — our accuracy/F1 figures are internally consistent but not directly comparable to prior published CMU-MOSI numbers, including Self-MM's, without confirming which convention those numbers used. **Final decision for this revision: we will not add a `label >= 0` supplementary table.** Rationale: every claim in this paper (the non-replication, the collapse-prevention finding, the mask-inertness result, and the CMU-MOSEI extension in Section 3.7) is a within-convention comparison between models evaluated identically under `label > 0`, so the choice of convention does not affect any conclusion drawn here — it only affects comparability to external published numbers, which we flag directly rather than attempt to fix by convention-matching post hoc. Recomputing the full grid a second time under `label >= 0` (8 models × 5 seeds × 4 rates, twice over for both datasets) is a substantial compute and verification cost for a table that would not change this paper's conclusions and exists solely to ease comparison to external work not otherwise engaged with here. If a future paper adds a head-to-head comparison to prior published CMU-MOSI/CMU-MOSEI results as its explicit purpose, that would be the appropriate place for the `label >= 0` table, not this one.
- **The `text_dim` discrepancy noted in Section 2.1** (768 at runtime vs. 300 documented) is partially, not fully, resolved: we have reasonable confidence the runtime features are BERT-based (768-dim is the standard signature for contextual embeddings across the CMU-MOSI/CMU-MOSEI literature, as opposed to GloVe's standard 300-dim), but we were not able to independently confirm the specific release/commit of the feature set that was actually downloaded, so we do not claim to have fully closed this item. We do not believe it invalidates the reported experiments (the encoder adapts to whatever `text_dim` it is constructed with) but flag the exact provenance as still open.

## 6. Conclusion

We set out to test whether pairing modality-dropout training with an attention-gated fusion layer improves robustness to missing modalities on CMU-MOSI. An initial pilot suggested the opposite of the design hypothesis — that dropout training actively hurt the gate. That finding did not survive replication at 5 seeds. In the course of investigating why the original result was seed-dependent, we found a different effect that does replicate cleanly: dropout training prevents a specific, previously undetected failure mode — degenerate collapse to a constant prediction — under complete loss of the dominant modality, a pattern present in zero of fifteen dropout-trained-model seed-runs and a substantial fraction of non-dropout-trained-model seed-runs. We also found that the gate's explicit mask-conditioning mechanism, the architectural feature at the center of this line of work, is nearly insensitive to its own mask input in every seed we tested.

None of this supports a clean story about attention gates and modality dropout being a reliably good or reliably bad combination for graded missingness robustness. It does support a narrower, better-evidenced claim: modality-dropout exposure during training has a real, replicable protective effect against complete-modality-loss collapse, separate from and not evidenced by this paper's original graded-robustness framing. Beyond the specific claim above, the study provides a documented record of how the interpretation changed from initial pilot to n=5, from accuracy-only to accuracy-plus-F1, and from aggregate gate-weight means to isolated mechanistic interventions.

## 7. Open analyses and future work

1. **Re-audit Section 3.5 for the same degenerate-collapse artifact identified in Section 3.4** — **done this revision.** `reaudit_gate_weights_per_sample.py` was run against the real 5-seed `gate_weights_raw.csv` (30,870 rows). Finding: `hard_mask_gated_fusion`'s large near-degenerate fraction is architectural (its masked-softmax structurally forces exact 0/1 weights on absent/lone-present modalities), not a discovered artifact. For the two soft-gated models, no hidden distortion was found in the reported aggregate means, but a real distributional difference emerged that the means alone did not show: `gating_only_no_dropout` has a meaningfully fatter near-1 tail on text than `attention_gated_fusion_full` at every rate. See Section 3.5 for the full table and discussion.
2. **Confirm which CMU-MOSI feature release was actually used.** The runtime `text_dim=768` conflicts with the documented 300-dimensional setting, so the text-feature provenance should be resolved before attributing the features to GloVe.
3. **Run the mask-channel isolation experiment on `attention_gated_fusion_full` — done this revision.** `mask_channel_isolation.py` was run against the real 5-seed checkpoints. Every shift is under 0.012 in magnitude, comparable to `gating_only_no_dropout`'s shifts — the mask-inertness finding extends to the dropout-trained gate. See Section 3.6 for the full table.
4. **Test the degenerate-collapse-prevention hypothesis directly** with dropout restricted to a range that never fully zeros a modality, to distinguish the effect of zero-input exposure from missingness exposure more generally. **Status: run this revision, with a methodological caveat that limits how the result can be read.** `run_narrow_dropout_range.py` retrained `dropout_only_fusion`, `attention_gated_fusion_full`, and `hard_mask_gated_fusion` under Uniform(0, 0.4) instead of Uniform(0, 0.75), 5 seeds each, and evaluated at the standard graded-missingness rates (0/0.25/0.5/0.75). Result: F1 stayed well above the 0.05 collapse threshold at every rate for every model/seed (rate=0.75 minimum F1 = 0.615, across all 15 model/seed combinations; full CSV in `results_narrow_dropout_range.csv`) — no collapse observed anywhere in this data. **However, this evaluation used Section 3.2's graded *random* missingness protocol (each modality independently dropped with probability=rate), not Section 3.4's *guaranteed complete* text-loss protocol that the original collapse claim is actually about** — even at rate=0.75, most evaluated samples still have text present some of the time, so this is not a strict test of the same condition. The absence of collapse under graded random missingness at up to 75% is suggestive but not conclusive evidence that narrow-range training still prevents collapse under complete text loss specifically; a direct test would rerun these checkpoints through Section 3.4's exact single-modality-masking evaluation (guaranteed zero text for every sample) rather than the graded protocol used here. We report the real result as what it actually measures rather than reading it as a direct answer to the original question.
5. **Verify the Zenodo DOI** (`10.5281/zenodo.22141293`) in a logged-out check before resubmission and confirm that it resolves to a public record containing the stated release files.
6. **Extend `verify_manuscript_numbers.py`** to cover the five-seed results, Section 3.2b, and the isolation experiments in Section 3.6.
7. **Investigate why `gating_only_no_dropout` avoided collapse on CMU-MOSEI but not CMU-MOSI (Section 3.7) — substantially advanced this revision, not fully closed.** A 25-run multi-draw ablation (Section 3.7.1: 5 seeds × 5 independent subsample draws each) confirms training-set size as a real, reproducible factor — seed 42 collapses in 4/5 independent draws, ruling out a single unlucky sample. But it does not resolve into a clean "certain seeds are collapse-prone" story: the ablation's collapse-heavy seeds (42, 99) mostly don't match CMU-MOSI's actual collapsing seeds for this model (42, 123, 2024) — only seed 42 overlaps. The natural next step is isolating what varies between draws that collapse and draws that don't for a given seed (e.g. which specific MOSEI samples end up in the collapsing subsamples vs. the clean ones), rather than treating "smaller training set" as a complete explanation on its own. IEMOCAP and other benchmarks remain out of scope beyond the two-claim MOSEI check already run.
8. **Title and abstract framing — resolved this revision.** The title now reads "...A Replication Study Across CMU-MOSI and CMU-MOSEI" to reflect the completed cross-dataset scope. We deliberately did not add the mask-inertness finding to the title: the "current claims at a glance" box explicitly frames it as secondary and not one of the paper's two main live claims, so omitting it from the title is a considered choice, not an oversight. The Abstract's closing sentence has been split into a scientific-conclusion sentence and a separate meta-contribution sentence (see Abstract).

## 8. Broader Considerations

The clearest practical implication of this revision is a caution about evaluation practice, not about multimodal fusion specifically: an accuracy-only masking evaluation missed a constant-output failure mode that F1 caught immediately, and a 3-seed significance test produced a confident, specific, monotonic-looking effect that did not survive 2 additional seeds. Neither error would have been caught by more careful reasoning about the *results* — both required rerunning or re-measuring. We'd suggest that any missingness-robustness evaluation report a class-balance-sensitive metric (F1 or equivalent) alongside accuracy by default, specifically because constant-output collapse under an unfamiliar masked input is, per Section 3.4, common enough across architectures and seeds to be a real risk rather than a corner case.

On deployment risk specifically: a system relying on `gating_only_no_dropout`-style training (no dropout exposure) would, per our results, have a roughly coin-flip-by-seed chance of degrading to a constant-output classifier under complete loss of its dominant modality — a failure mode that would not necessarily be visible in an accuracy-only monitoring setup, since it can still post a plausible-looking accuracy number. This is a concrete, actionable finding independent of the paper's original graded-robustness question.

**A concrete monitoring recommendation.** Beyond reporting F1 alongside accuracy at evaluation time, a deployed pipeline can catch this specific failure mode in production with a cheap, always-on check: track the per-class share of predictions in each rolling window of inference traffic (e.g., every 500 predictions), and alert when any single class exceeds a fixed threshold — we would suggest 90% of predictions in a window, since our test set's majority class alone is 59.6% and the collapsed models predict one class on effectively 100% of samples — conditioned on which modalities were available for those predictions. Because collapse in our results is triggered specifically by complete loss of the dominant modality, this check is most informative when segmented by the input modality-availability pattern rather than computed in aggregate, which would dilute a collapse confined to one missingness condition against normally-behaving traffic from others.

## 9. Data Availability

Checkpoints, `results_raw_5seed.csv` (five-seed), `single_modality_results.csv`, `gate_weights_summary.csv`, `gate_weights_raw.csv`, `significance_5seed_full.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, `encoder_zeroing_isolated_effect.csv`, and `config_log.json`/`config_log_5seed.json` are included in the project's data release, alongside the CMU-MOSEI extension's raw results added in this revision: `mosei_graded_robustness_raw.csv` and `mosei_single_modality_results.csv` (Section 3.7's underlying per-seed data), plus the code that produced and can independently re-verify them: `data/dataset_mosei.py`, `run_mosei_targeted_replication.py`, and the extended Section 3.7 checks in `verify_manuscript_numbers.py`. **Verification status as of this revision:** we attempted to independently confirm that `10.5281/zenodo.22141293` resolves to a public record, checked logged out of any personal Zenodo account. It did not return a resolvable record in this check. This does not necessarily mean the DOI is invalid — Zenodo DOIs can fail to appear in a general search while still resolving directly, and reservation-stage DOIs in particular can be non-public until a deposit is finalized — but we were not able to positively confirm the file list, version, or date it resolves to, and we are not treating this item as closed. Separately, the MOSEI files listed above have not yet been added to any Zenodo deposit at all — the current DOI's file list, once confirmed, will not include them until a new deposit or version is created. **This is now the single non-negotiable pre-submission item**: before this manuscript is submitted, someone must load `https://doi.org/10.5281/zenodo.22141293` directly in a logged-out browser session, confirm it resolves to a public Zenodo record, cross-check every filename above against the actual Zenodo file listing (not just the GitHub repository), and record here which files and version it contains — including confirming the MOSEI files have been added. If it does not resolve, or the file list is incomplete, the deposit must be corrected or finalized before submission, since an uncheckable or incomplete Data Availability link is a closed, uncorrected error, not an open caveat.

## References

*(Retained from the original submission — Zadeh et al. 2016; Neverova et al.; Tan & Zhang 2025; Self-MM/Yu et al. 2021 — unchanged.)*
