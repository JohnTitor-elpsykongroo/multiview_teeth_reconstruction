# R2 evaluation and deployment protocol audit

This is a research diagnosis and a plan for subsequent checks, not R3/R4 approval.
The active effective-count control remains frozen at
`17f8c07c3a1a582b3916a5fff6e6d74b855bd983`; do not merge changes while its driver
is active. Preserve its common fitting objective and all native results.

The completed parent is `runs/r2_baseline_20261006T134608Z`, trained at `4fd0d5a`.
Its 32 learned-train and four val cases per arch all produced valid geometry.
Full numerical evidence and checkpoint hashes are in
`runs/r2_effective_20261006T1634Z/development_evidence/R2_baseline_review_20261006T1622/R2_baseline_decision.json`.
The active plan's `parent_review/` link is a prose path error; the driver argument
and frozen inventory bind the actual `development_evidence/` path above.

## Actual code contracts

| Property | Geometric auto-decoder evaluation | Image reconstruction bundle |
| --- | --- | --- |
| Decoder | Frozen | Frozen |
| Tooth codes | Raw codes fitted from original present trained-row means | `mu + L @ q`, with train-only covariance |
| Gum | Temporary fitted raw code | Fixed original trained-row mean |
| Prior | `weights.latent * mean(code**2)` within the training recipe | `prior_weight * mean(q**2)` |
| Observation | Disjoint fit/evaluation scan points and tooth centers | Known-camera semantic image observations |
| Current steps | 20 Adam steps, lr .001 | Staged fitting configuration, a different inference task |

Sources: `training/evaluation.py:evaluate_cases`,
`training/arch_training.py:ArchTrainingSystem.loss` and `export_bundle`,
`dmm/bundle.py:ModelBundle.codes`, `dmm/scene.py:latent_prior`,
`dmm/fitting.py:fit_scene` (under `third_party/DMM`). Covariance export uses
95% empirical covariance + 5% diagonal shrinkage, then an eigenvalue floor of
`1e-6 * trace(covariance)/dimension`. It does not impose a fixed absolute variance.

Consequences: R2 geometric val is a legitimate temporary-code auto-decoder
measurement under the guide; it does not establish image fitting quality or the
fixed-gum deployment result. Raw-code and standardized-q priors are different
objectives. Small empirical code variance or small fixed-point response alone
does not demonstrate functional collapse. No representation or bundle contract
change is proposed by this audit.

## Diagnostic sequence after the active control

First inspect every matched-epoch common comparison, native failures, tooth mean
and point P95, learned-train/val gap, area/topology, clipping and resource budget.
Do not pick the best native effective score as an isolated training-factor result.
If a material gain exists, check its teeth and cases against mesh sampling spread
and repeat the important comparison before expanding to R3.

Select fixed val representatives before subsequent evaluation: first declared
case `01343APK` and shared val case `0145NKWZ` per arch, retaining all their
present teeth. Use the baseline and a valid candidate checkpoint at matched
epochs; checkpoint choice and hashes must be frozen in a fresh plan before launch.
Do not use test data or change source while the current job is active.

1. Capture exact post-fit codes and evaluated points under the existing common
   recipe. New evaluation artifacts preserve `codes.npz`, with `code_<label>`
   arrays and `model_unit_mm=50`; `field.json` binds its SHA256. Failure artifacts
   preserve the same codes when field geometry is invalid. Old artifacts lacking
   codes must receive a fresh, explicitly recorded evaluation; never reconstruct
   val codes from a train embedding or claim an old fit was recovered exactly.
2. With these codes, observations, checkpoint and sampling seed frozen, reproduce
   grid48 distances and tooth metrics first. Then re-extract grid64 and grid80
   surfaces. Stop on nonfinite fields, nonpositive domain boundaries or missing
   teeth, and retain all failures. Apply the same budgets to both candidates.
   Grid changes also alter area sampling, so use the five declared mesh seeds
   when interpreting a difference near the fixed-mesh seed spread.
3. Separately compare fresh 20/80-step fits on the same fit/evaluation partitions,
   same initialization, learning rate and common objective; change steps only.
   Freeze evaluation point indices, mesh resolution and sampling budget. Report
   per-case/tooth changes and convergence, including worsening and failures.
4. Separately freeze gum to its original present trained-row mean while holding
   the tooth fit recipe and observations fixed. Refit tooth codes independently
   for each condition; a post-fit gum swap is only a sensitivity diagnosis.
   Standardized-q fitting is a further independent protocol change, not an
   equivalent parameterization with the same Adam steps and prior coefficient.

Proposed initial diagnostic cap: four cases (two per arch), at most two candidate
checkpoints per arch, one GPU driver, 3600 monotonic seconds and 40GiB disk reserve;
fixed-code CPU extraction instead gets its own 900-second cap, eight CPU threads,
and no CUDA access. Freeze the actual plan, measured cost, source/config/data/code/
point/checkpoint hashes and stop rules before execution. These caps are proposals
for a subsequent experiment, not extensions to the active R2 control's 10800s cap.

R3/R4 quality decisions require valid geometry for the entire declared cohort,
reviewed per-tooth tails and representative views, improvements supported by
repeatability evidence, and a common protocol applied to old and new candidates.
Do not invent a clinical millimeter threshold, discard hard cases, shift the iso
surface, or approve a model only because a mean score improves. Freeze any
revised selection protocol before R3/R4; keep the original measurements distinct.
R4 final test remains sealed until checkpoint and protocol selection are frozen.
