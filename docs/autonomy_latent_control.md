# Optional R2 latent regularization control

`configs/autonomy/r2_latent_1e5/{upper,lower}.json` changes only
`recipe.weights.latent`: baseline 1e6 to 1e5. The 32 train/four-val cases, seeds,
point sampling, reduction, clipping, initialization, learning rates, scheduler,
checkpoint epochs and native evaluation budget remain identical. This is a fresh
training option, not a cross-recipe resume, and has not been launched.

R2 upper CPU probes found shrinking original-present code variance and weak fixed
learned-train field/warp response at epoch15. They support asking whether latent
regularization limits patient variation; they neither identify its cause nor show
that this candidate will improve mesh quality. The original paper/upstream uses
1e6; the candidate is explicitly a project experiment, not a claimed paper fix.

First complete the active baseline's full train/val evaluation and the prepared
effective-denominator control. A persistent weak code response together with mesh
or patient adaptation evidence may justify this separate baseline-reduction
control. Do not combine it with effective reduction or clipping changes in its
first comparison. Freeze commit/config/data/checkpoint SHA and a measured resource
budget, run fresh preflight and real small native/common evaluator gates, then
start only after the prior GPU driver has fully exited.

For decoder comparisons, reevaluate matched checkpoint epochs5/10/15/20 using the
same `configs/autonomy/evaluation_fit_v1` recipe and budgets as baseline. Keep native
best-val diagnostics separately: changing training regularization also changes its
default latent-fitting objective. Report all failures, equal-case/tooth geometry,
tooth tails, train/val gaps and fixed-point code response. Do not rank cross-recipe
total losses. A common fitting recipe's ability to adapt patient codes requires
its own fitting-budget diagnosis before final protocol selection.

Stop on data/source/binding drift, nonfinite values, failed sampling/evaluator
gates, OOM or the frozen resource cap; retain recovery checkpoints and failures.
Do not enter R3/R4 from scalar means alone. Final test remains unopened until R4
selection and checkpoint/protocol freeze.
