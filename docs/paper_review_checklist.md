# Paper Review Checklist

## Introduction
- [ ] Problem/gap stated in the first 1–2 paragraphs, not buried
- [ ] Prior work cited specifically (not just "prior work has shown...")
- [ ] Your contribution stated as a distinct claim, separable from adopted/known techniques
- [ ] Scope is bounded (what this paper does *not* claim)

## Methods
- [ ] Dataset(s) named, with size, source, and license/access terms
- [ ] Preprocessing steps fully specified (enough to reproduce)
- [ ] Model architecture described with enough detail to reimplement (layer sizes, gating mechanism, training regime)
- [ ] Baselines named explicitly, including at least one recent (post-2023) comparison
- [ ] Hyperparameters and training setup listed (optimizer, LR, epochs, batch size, hardware)
- [ ] Missingness/dropout simulation protocol specified (how modalities are dropped, at what rate, train vs. test)

## Results (emphasis)
- [ ] Every claimed number traceable to a specific table/figure — no unsupported numbers in prose
- [ ] Main metric(s) reported with variance (std dev or CI across seeds/runs), not single-run point estimates
- [ ] Statistical significance tested where comparisons are close (not just "our method is better")
- [ ] Efficiency numbers included: parameter count, FLOPs, inference latency — actually measured, not asserted
- [ ] Ablation table isolating the gating mechanism's contribution vs. dropout training alone
- [ ] Results broken down by missingness pattern/rate, not just one aggregate number
- [ ] Comparison against the post-2023 baseline is present, not only classical fusion baselines
- [ ] Failure cases or negative results acknowledged, not omitted
- [ ] All plots/tables generated reproducibly from the Colab notebook (seed set, code cell that regenerates each figure)
- [ ] Random seed(s) fixed and reported for every reported run

## Discussion
- [ ] Results interpreted, not just restated
- [ ] Explicit comparison back to the novelty claim from the Introduction — did the gating mechanism actually deliver what was promised?
- [ ] Limitations section present (dataset scope, modality types tested, compute constraints)
- [ ] Alternative explanations for results considered (e.g., is the gain from gating or from dropout training alone?)

## Conclusion
- [ ] Contribution restated precisely, matching (not exceeding) what Results actually support
- [ ] Generalization/cross-domain claims softened to "hypothesized" unless a second-domain experiment was run
- [ ] Future work section distinct from unfinished current work
- [ ] No new claims introduced that weren't in Results/Discussion
