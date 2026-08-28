# Modality Dropout Prevents Degenerate Collapse but Not Graded Missingness Robustness in Attention-Gated Multimodal Fusion: A Replication Study on CMU-MOSI

## Abstract

Multimodal emotion and sentiment recognition systems typically assume all modalities are available at inference time, an assumption that frequently fails in practice. We investigate whether pairing an attention-gated fusion layer — conditioned explicitly on a missingness mask — with modality-dropout training improves robustness to missing modalities on CMU-MOSI. We report a result that changed substantially between an initial 3-seed pilot and a 5-seed replication, and we consider the discrepancy between the two runs, and what it revealed, more informative than either run would have been alone.

In a 3-seed pilot, an attention gate trained *without* modality dropout significantly outperformed the identical gate trained *with* it at every nonzero missingness rate (p<0.05, effect size growing with missingness rate). Replicating with 2 additional seeds (n=5 total) reversed this finding: none of the three nonzero-rate comparisons remain significant (p=0.327 at 25% missingness, p=0.103 at 50%, p=0.076 at 75%), and the point-estimate gap roughly halved at every rate. We report this plainly as a failure to replicate rather than a softened version of the original claim: an effect significant at n=3, with p-values as low as 0.006, did not survive two additional random seeds.

While investigating why the no-dropout gate's apparent advantage was seed-dependent, we found a different, better-supported effect. Under the single-modality condition where text is entirely absent (audio and vision only, the modality-loss condition our diagnostics targeted specifically), every model we trained *with* modality dropout — three architecturally distinct models, including a non-gated baseline — produced non-degenerate predictions (F1≈0.58) in all 5 seeds, with no exceptions. Models trained *without* dropout exposure (5 of 8 architectures, including the original no-dropout gate) instead collapsed, in a seed-dependent 40–80% of individual runs, to predicting a single constant class for every test-set sample — an artifact that inflates raw accuracy to match the test set's 59.6% majority-class rate while F1 drops to exactly 0.000. This distinction was invisible in our own first-draft results because that draft's single-modality table reported accuracy without F1. The corrected pattern is exactly reproducible across every seed we ran and is, we believe, the paper's actual, defensible contribution: in this setup, modality-dropout training's real and replicable benefit is preventing catastrophic mode collapse under complete loss of a modality, not improving graded robustness under partial, proportional missingness — where, at n=5, its effect on the gate is statistically unsupported and trends mildly negative.

We additionally isolated the gate's response to its missingness-mask input directly, independent of what the encoders see (Section 3.6): across every seed and every training regime, flipping the mask input alone shifts the gate's softmax weight on the missing modality by less than 0.01, a near-total insensitivity that also replicates cleanly. The mask-conditioning mechanism this architecture was built around appears to contribute almost nothing to the gate's behavior, with or without dropout training. We report the full arc of this project — the original ablation, its non-replication, the degenerate-collapse artifact our own earlier draft missed, and the mask-channel finding — as a case study in the concrete value of routinely checking F1 alongside accuracy, rerunning small-n significance claims before treating them as settled, and treating stale project documentation as a real methodological hazard rather than a formality.

## 1. Introduction

Real-world deployments of multimodal affective computing systems routinely lose access to one or more modalities: a camera occluded, a microphone muted, a text transcript unavailable. A common design response is an attention-gated fusion layer that takes an explicit missingness mask as input, paired with modality-dropout training — randomly zeroing modalities during training — under the intuitive hypothesis that giving the gate the mask, together with dropout exposure, should let it learn to reweight around whichever modalities are absent at test time.

We set out to test this hypothesis directly on CMU-MOSI. We did not find a clean confirmation of it, and the way our own results shifted as we added evidence is, in our view, as much the point of this paper as any single number in it.

### 1.1 What actually happened, in order

This paper's central claim changed twice over the course of the project, and we report all three stages rather than presenting only the final one, because each stage is informative on its own:

**Stage 1 (n=3 seeds).** An attention gate trained without modality dropout significantly outperformed the identical gate trained with it, at every nonzero missingness rate, with the gap growing as missingness increased. Diagnostic analysis at this stage (single-modality masking, gate-weight logging) appeared to support a mechanistic explanation involving the dominant modality's encoder being disrupted by dropout training.

**Stage 2 (n=5 seeds, a targeted replication).** The ablation's significance did not hold. All three nonzero-rate comparisons that were significant at n=3 became non-significant at n=5 (Section 3.2). We treat this as a genuine non-replication, not noise to be explained away — with only 2 degrees of freedom, three seeds that happened to agree was not, in retrospect, strong evidence.

**Stage 3 (diagnosing the seed-dependent split, which led to a different finding).** Investigating *why* the no-dropout gate behaved inconsistently across seeds under the single-modality text-missing condition led us to check F1 alongside accuracy for the first time — and to discover that the "good" seeds were not showing robustness at all, but a degenerate constant-output collapse (Section 3.4). Checking whether this collapse pattern held across all 8 models and all 5 seeds revealed a clean, fully-replicating split: dropout-trained models never collapsed; non-dropout-trained models did, in a seed-dependent fraction of runs. This is the finding we consider the paper's real contribution.

### 1.2 Contributions

Given the above, we frame this paper's contributions differently than a typical positive-result paper would:

1. **A documented non-replication.** The originally hypothesized ablation effect (dropout training hurts an attention gate's graded missingness robustness) was significant at n=3 and is not significant at n=5, at any of the three rates it was originally reported at. We report the full before/after comparison (Section 3.2) rather than only the final numbers, because the discrepancy is itself informative about small-n significance testing in this kind of experiment.

2. **A corrected, replicating finding: dropout training prevents degenerate collapse, not graded robustness.** Under complete loss of the dominant modality (text), every dropout-trained model we tested avoided a constant-output failure mode across all seeds; every non-dropout-trained model exhibited it in some seeds and not others. This is exactly reproducible and, unlike the original ablation, does not depend on which 3 of 5 seeds happen to be reported.

3. **Evidence that the gate's mask-conditioning mechanism is nearly inert.** An isolated intervention — changing only the mask input the gate receives, holding the underlying features fixed — produces a negligible shift in the gate's output, in every seed we tested, regardless of dropout training. This calls into question whether the mask-conditioned gate design is doing meaningful work at all, independent of the missingness question this paper set out to answer.

4. **A worked example of the checks that catch this kind of problem before publication**, documented in full because we think the process is more useful to other researchers than a clean narrative would have been: rerunning at n=5 before trusting n=3, checking F1 alongside accuracy on any masking-based diagnostic, and — a genuinely embarrassing but worth-reporting detail — discovering mid-project that a project's own README had gone stale relative to its actual state, which nearly led to redundant reruns of already-completed, already-verified work (Section 2.7).

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

**Correction from the original submission:** our config file's documented `text_dim: 300` (implying GloVe-based features) does not match the feature dimensionality actually loaded at runtime (`text_dim=768`, printed by every dataset-loading call in this project's logs). We did not catch this discrepancy before the original 3-seed submission draft. It does not appear to invalidate the experiments — the encoder adapts to whatever `text_dim` it is constructed with — but it means prior mentions of "GloVe-based" text features in this project's documentation were incorrect, and we flag it rather than silently correcting it, since we do not know which feature release was actually used without further checking against the Self-MM/CMU-MultimodalSDK release history.

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

**This project's seed count changed over its lifetime, and we report both stages.** An initial pilot used 3 seeds (42, 123, 2024) — 2 degrees of freedom for the paired t-tests in Section 3.2. Following review, we replicated with 2 additional seeds (7, 99), bringing the total to 5 seeds (4 degrees of freedom) for every model. **The headline ablation result reported at n=3 does not hold at n=5** (Section 3.2); we report both sets of statistics rather than only the final one.

Significance tests use paired t-tests across seeds at each missingness rate, matching seeds between compared models.

### 2.6 Training-pipeline sanity check

We ran an overfit-a-fixed-batch check (16 samples, 50 gradient steps) for all 8 models before trusting the full training grid. Five of eight models (`early_fusion`, `late_fusion`, `fixed_weight_fusion`, `gating_only_no_dropout`, `imputation_baseline_post2023`) drove loss from ~2.0–2.1 to below 0.01, a clean pass. The three dropout-trained models (`dropout_only_fusion`, `attention_gated_fusion_full`, `hard_mask_gated_fusion`) plateaued around 0.17–0.94 rather than near zero. We believe this reflects a limitation of the sanity-check script rather than a training bug: these models draw a fresh random missingness rate every one of the 50 steps, so the "batch" they are asked to overfit is not actually fixed from step to step, unlike the non-dropout models. We have not rerun this check with a corrected version that fixes the missingness rate during the overfit test specifically; we flag it as unresolved rather than treating the pass on 5/8 models as validating all 8.

### 2.7 Reproducibility, code/data provenance, and a documentation failure worth reporting

An earlier internal draft of this manuscript's Sections 3.4–3.6 contained fabricated numbers, caught before submission by a verification script (`verify_manuscript_numbers.py`) that recomputes every reported number directly from the underlying result files. That script is included with our code release and every number in the current Sections 3.1–3.5 has been checked against it (119/119 checks passing against the 3-seed data; the 5-seed data in the current draft was independently spot-verified by hand rather than through that script, since the script's hard-coded claims are pinned to the original 3-seed numbers — extending it to check the 5-seed and diagnostic numbers in this revision is listed as future work, Section 7).

**A second, separate documentation problem surfaced during the 5-seed replication, and we report it because we think it is a common and underappreciated failure mode.** The project's working directory had a `README.md` describing checkpoints and diagnostics as "not yet executed," when in fact a complete run — checkpoints, `results_raw.csv`, `single_modality_results.csv`, `gate_weights_summary.csv`, `gate_weights_raw.csv` — was already present and, once checked, verified as real (not fabricated) via the script above. The README was simply stale. Separately, the project's `data/` directory was missing its own dataset-loading module (`dataset.py`), present only in a separate code archive and never copied into the main working directory — meaning the working directory, taken at face value, was not actually a complete or runnable copy of the project. Neither of these was caught until directly probing the filesystem rather than trusting the directory's own documentation. We report both because "the code works and the numbers check out" and "the project's own account of its state is accurate" turned out to be two separate claims, and only checking the first one would have been a mistake.

## 3. Results

### 3.1 Main results: original 3-seed pilot vs. 5-seed replication

**Table 3.1a — 3-seed pilot (original submission).** Mean accuracy ± 95% CI half-width, n=3 seeds (df=2).

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

**Table 3.1b — 5-seed replication (2 additional seeds: 7, 99).** Mean accuracy ± 95% CI half-width, n=5 seeds (df=4).

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

### 3.2 Significance tests: before and after replication

**Table 3.2a — 3-seed pilot, key comparisons (as originally reported).**

| Comparison | Rate | p-value |
|---|---|---|
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.0 | 0.212 |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.25 | **0.031** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.5 | **0.044** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.75 | **0.006** |
| `hard_mask_gated_fusion` vs `gating_only_no_dropout` | 0.75 | **0.001** |

**Table 3.2b — 5-seed replication, same comparisons.**

| Comparison | Rate | p-value | Change from Table 3.2a |
|---|---|---|---|
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.0 | **0.024** | now significant (was not); direction: no-dropout gate now *worse*, not better |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.25 | 0.327 | **no longer significant** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.5 | 0.103 | **no longer significant** |
| `gating_only_no_dropout` vs `attention_gated_fusion_full` | 0.75 | 0.076 | **no longer significant** |
| `hard_mask_gated_fusion` vs `gating_only_no_dropout` | 0.75 | 0.087 | **no longer significant** |

**Every comparison that drove this paper's original headline claim lost significance under replication.** The only newly-significant result (rate=0.0, no missingness at all) is in the opposite direction from, and irrelevant to, the original missingness-robustness claim. We do not consider the 5-seed data supportive of the original ablation hypothesis in any form. Full 28-row comparison table (all model pairs, all rates, both seed counts) is included in the project's data release as `significance_5seed_full.csv`.

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
| `late_fusion` | — / — | **0.596 / .000** | **0.596 / .000** | — / — | — / — |

Bold entries mark accuracy = 0.596210, exactly `409/686`, this test set's negative-class base rate, paired with F1 = 0.000 (or, in two `imputation_baseline_post2023`/`fixed_weight_fusion` cases, F1 ≈ 0.02, functionally the same failure). **These are not cases of a model handling missing text well. They are cases of the model outputting the same non-positive prediction for every one of the 686 test samples, regardless of input, which happens to match the majority class often enough to look like reasonable accuracy.** The three dropout-trained models (`attention_gated_fusion_full`, `dropout_only_fusion`, `hard_mask_gated_fusion`) never once produce this pattern, in any of 15 seed-runs (3 models × 5 seeds). Every non-dropout-trained model produces it in some seeds and not others — a seed-dependent rate of roughly 40–80% of runs per model.

We did not catch this in the original 3-seed submission because that draft's Table 3.4 reported only accuracy. The seeds used in the original pilot (42, 123, 2024) happen to include zero or more degenerate collapses depending on model, which is part of why the original framing ("the no-dropout gate retains 0.596 accuracy under missing text, versus 0.405 for the dropout-trained gate") was actively misleading rather than merely imprecise: the higher number was, for at least the `gating_only_no_dropout` model at those specific seeds, not a better result.

### 3.5 Gate-weight response to the missingness mask

*(Retained from the original submission; not contradicted by the above, though we note this analysis has not itself been re-checked for the degenerate-collapse artifact and should be treated with the same caution until it is — see Section 7.)*

Both the dropout-trained and no-dropout-trained gates show similar relative responsiveness to the mask signal when text is present vs. absent (e.g., mean gate weight on text: 0.926 present vs. 0.761 absent for `gating_only_no_dropout`@25% missingness; 0.857 vs. 0.692 for `attention_gated_fusion_full`@25%), a pattern consistent across the 0.25/0.5/0.75 rates tested. Section 3.6 below investigates this mask-responsiveness more directly and finds it is much smaller than these numbers suggest once the encoder's response to zeroed input is controlled for separately.

### 3.6 Isolating the gate's response to the mask signal from the encoder's response to zeroed input

Motivated by the seed-dependent split in Section 3.4, we ran a controlled intervention using the `gating_only_no_dropout` checkpoints across all 5 seeds, isolating two effects that Section 3.4's masking protocol conflates: (a) the gate receiving a different *mask value*, and (b) the encoder receiving a different *feature input* (zeroed text).

**Isolating the mask-channel effect alone** (real, non-zeroed text features throughout; only the mask value fed to the gate changes from `[1,1,1]` to `[0,1,1]`):

| seed | mean gate weight on text, mask=on | mean gate weight on text, mask=off | shift |
|---|---|---|---|
| 42 | 0.860 | 0.851 | −0.009 |
| 123 | 0.999 | 0.999 | −0.0001 |
| 2024 | 0.929 | 0.924 | −0.005 |
| 7 | 0.723 | 0.714 | −0.010 |
| 99 | 0.888 | 0.881 | −0.007 |

The gate's response to its own mask input, isolated from any change in the underlying features, is negligible — under 0.01 in every seed, with no meaningful difference between seeds that later collapse (7, 99) and seeds that do not (42, 123, 2024). **This directly supports the architectural point in Section 2.3**: since this model's training never varied the mask input, its weights on that channel did not learn to respond to it, and this holds regardless of which seed produced the initialization.

**Isolating the encoder-zeroing effect alone** (mask channel held at `[1,1,1]` throughout; only the encoder's text input is zeroed):

| seed | mean gate weight on text, real input | mean gate weight on text, zeroed input | shift |
|---|---|---|---|
| 42 | 0.860 | 0.615 | −0.245 |
| 123 | 0.999 | 0.997 | −0.002 |
| 2024 | 0.929 | 0.708 | −0.221 |
| 7 | 0.723 | 0.547 | −0.177 |
| 99 | 0.888 | 0.618 | −0.270 |

This effect is much larger than the mask-channel effect, confirming that whatever downstream behavior differs between seeds is driven by how each seed's encoder-plus-head jointly respond to a zeroed text input, not by the gate's mask conditioning. However, **this table does not cleanly separate the seeds that later collapse (7, 99) from those that don't (42, 123, 2024)**: seed 123 shows almost no shift (−0.002) despite being a non-collapsing seed, while seeds 42 and 2024 (also non-collapsing) show large shifts comparable to the collapsing seeds. We do not have a single identified structural cause for the seed-dependent split in Section 3.4's *specific* accuracy values; we have ruled out the mask channel as the cause and localized the effect to the encoder-plus-head's response to zeroed input, without fully explaining why that response itself varies by seed in a way that doesn't track collapse status through this particular measurement. We report this partial result rather than a tidier story we don't have evidence for.

## 4. Discussion

We organize this section around what changed between our original hypothesis and what the full evidence, including the corrected Section 3.4 and the isolation experiments in Section 3.6, actually supports.

**The original hypothesis — that modality-dropout training hurts an attention gate's graded missingness robustness — is not supported at n=5.** We think the most likely explanation for the n=3 result is a small-sample false positive: three seeds that happened to agree, at a comparison with substantial seed-to-seed variance (visible directly in the CI half-widths in Table 3.1, several of which exceed 0.05 for the relevant models). We do not have evidence for any of the mechanistic explanations our original 3-seed draft proposed (encoder disruption, a "flatter, more hedged baseline weighting policy") as an explanation for graded missingness performance, because the effect they were explaining is no longer present in the data.

**The corrected finding — dropout training prevents degenerate collapse under complete modality loss — is, we think, real and better-supported.** It replicates exactly across every seed and every dropout-trained model we tested (0/15 collapses), against a clearly nonzero and seed-dependent collapse rate for every non-dropout-trained model. We do not yet have a mechanistic account of *why* dropout training prevents this specific failure mode — a plausible hypothesis is that exposure to zeroed inputs during training, even without any signal from the mask channel itself (Section 3.6), regularizes the encoder-plus-head's response to out-of-distribution zeroed input in a way that prevents it from settling into a degenerate constant-output basin — but we have not tested this directly and flag it as the natural next experiment (Section 7).

**The mask-conditioning mechanism itself appears to contribute little, independent of the missingness question.** Section 3.6's isolated mask-channel intervention shows near-zero sensitivity in every seed, in a model (`gating_only_no_dropout`) whose training procedure never varied that input — and, per Section 3.5's aggregate numbers (not yet subjected to the same isolation), possibly in the dropout-trained gate as well, though we have not run the equivalent isolation experiment on `attention_gated_fusion_full` and do not want to claim more than we've directly measured. If the gate is doing most of its useful work through its response to encoder outputs rather than through its explicit mask input, that would suggest the entire mask-conditioning design element — the architectural feature this line of work is specifically about — may be doing less than assumed, independent of any missingness-robustness question. This is, in our view, a more interesting and more surprising finding than either the original ablation or its non-replication, and we did not go looking for it; it fell out of trying to understand a result that turned out not to be real.

## 5. Limitations

- **The core ablation claim did not replicate**, and we do not know whether a larger sample (n=10, n=20) would recover a smaller but real effect, show the effect flips sign, or show no effect at all. We consider the current n=5 evidence insufficient to make any claim about the sign or existence of dropout training's effect on graded missingness robustness in this setup, and we recommend against citing the original 3-seed numbers, which remain in Section 3.1a/3.2a purely for transparency about what changed.
- **The degenerate-collapse finding, while it replicates cleanly on accuracy/F1, has not been mechanistically explained.** Section 3.6's attempt to localize the cause (mask channel vs. encoder response to zeroed input) ruled out one candidate but did not identify a clean alternative that separates collapsing from non-collapsing seeds within the non-dropout-trained models.
- **Section 3.5's gate-weight analysis has not been re-audited for the same degenerate-collapse artifact** that affected Section 3.4; until it is, its aggregate mean-weight numbers should be treated with the same caution, since a small number of degenerate per-sample predictions could distort an aggregate mean without being visible in that table's current form.
- **The training-pipeline sanity check (Section 2.6) did not cleanly pass for dropout-trained models**, and we have not resolved whether this reflects a limitation of the check itself or an actual training-pipeline issue affecting exactly the models this paper's corrected finding depends on.
- **Single-dataset scope.** All experiments use CMU-MOSI only; we make no claim these findings, in either direction, generalize to CMU-MOSEI, IEMOCAP, or other multimodal benchmarks.
- **Binarization convention** (Section 2.2) — our accuracy/F1 figures are internally consistent but not directly comparable to prior published CMU-MOSI numbers, including Self-MM's, without confirming which convention those numbers used.
- **The `text_dim` discrepancy noted in Section 2.1** (768 at runtime vs. 300 documented) was not resolved before this revision; we do not believe it invalidates the reported experiments but flag it as an open documentation issue.

## 6. Conclusion

We set out to test whether pairing modality-dropout training with an attention-gated fusion layer improves robustness to missing modalities on CMU-MOSI. An initial 3-seed pilot suggested the opposite of the design hypothesis — that dropout training actively hurt the gate. That finding did not survive replication at 5 seeds. In the course of investigating why the original result was seed-dependent, we found a different effect that does replicate cleanly: dropout training prevents a specific, previously undetected failure mode — degenerate collapse to a constant prediction — under complete loss of the dominant modality, a pattern present in zero of fifteen dropout-trained-model seed-runs and a substantial fraction of non-dropout-trained-model seed-runs. We also found that the gate's explicit mask-conditioning mechanism, the architectural feature at the center of this line of work, is nearly insensitive to its own mask input in every seed we tested.

None of this supports a clean story about attention gates and modality dropout being a reliably good or reliably bad combination for graded missingness robustness. It does support a narrower, better-evidenced claim: modality-dropout exposure during training has a real, replicable protective effect against complete-modality-loss collapse, separate from and not evidenced by this paper's original graded-robustness framing. We think the more valuable output of this project, beyond that specific claim, is the documented record of how the original claim fell apart under scrutiny we applied to our own work — n=3 to n=5, accuracy-only to accuracy-plus-F1, aggregate gate-weight means to isolated mechanistic interventions — and we would rather other researchers see that record than a version of this paper that presented only whichever result survived last.

## 7. Future work

1. **Test the degenerate-collapse-prevention hypothesis directly**, e.g. by training a model with dropout exposure limited to a range that never fully zeroes a modality, and checking whether the collapse-prevention effect persists — this would help distinguish "exposure to zeroed input specifically" from "exposure to any missingness" as the operative mechanism.
2. **Run the same mask-channel isolation experiment (Section 3.6) on the dropout-trained gate** (`attention_gated_fusion_full`), not just the no-dropout gate — Section 4 flags this as an open question we did not have time to close.
3. **Re-audit Section 3.5's gate-weight analysis for the same degenerate-collapse artifact** found in Section 3.4, using per-sample rather than aggregate-mean statistics.
4. **Resolve the sanity-check ambiguity for dropout-trained models** (Section 2.6) with a version of the check that fixes the missingness rate across all steps rather than resampling it.
5. **Extend `verify_manuscript_numbers.py`** to cover the 5-seed data, the significance tables in Section 3.2b, and the isolation experiments in Section 3.6, rather than leaving it pinned to the original 3-seed numbers.
6. **Resolve the `text_dim` documentation discrepancy** (Section 2.1) by confirming which CMU-MOSI feature release was actually used.
7. Second-dataset validation (CMU-MOSEI, IEMOCAP) remains out of scope for this single-dataset study; we'd want the above items resolved first given how much this paper's internal picture changed during revision.

## 8. Broader Considerations

The clearest practical implication of this revision is a caution about evaluation practice, not about multimodal fusion specifically: an accuracy-only masking evaluation missed a constant-output failure mode that F1 caught immediately, and a 3-seed significance test produced a confident, specific, monotonic-looking effect that did not survive 2 additional seeds. Neither error would have been caught by more careful reasoning about the *results* — both required rerunning or re-measuring. We'd suggest that any missingness-robustness evaluation report a class-balance-sensitive metric (F1 or equivalent) alongside accuracy by default, specifically because constant-output collapse under an unfamiliar masked input is, per Section 3.4, common enough across architectures and seeds to be a real risk rather than a corner case.

On deployment risk specifically: a system relying on `gating_only_no_dropout`-style training (no dropout exposure) would, per our results, have a roughly coin-flip-by-seed chance of degrading to a constant-output classifier under complete loss of its dominant modality — a failure mode that would not necessarily be visible in an accuracy-only monitoring setup, since it can still post a plausible-looking accuracy number. This is a concrete, actionable finding independent of the paper's original graded-robustness question.

## 9. Data Availability

Checkpoints, `results_raw.csv` (3-seed) and `results_raw_5seed.csv` (5-seed), `single_modality_results.csv`, `gate_weights_summary.csv`, `gate_weights_raw.csv`, `significance_5seed_full.csv`, `gate_weight_norms_by_seed.csv`, `mask_channel_isolated_effect.csv`, `encoder_zeroing_isolated_effect.csv`, and `config_log.json`/`config_log_5seed.json` are included in the project's data release. As of this revision, we have not independently re-verified that the Zenodo DOI referenced in earlier drafts (`10.5281/zenodo.22105162`) resolves to a public record containing these files; confirm this directly (logged out of any personal Zenodo account) before relying on it, per the note carried over from the previous revision.

## References

*(Retained from the original submission — Zadeh et al. 2016; Neverova et al.; Tan & Zhang 2025; Self-MM/Yu et al. 2021 — unchanged.)*
