# Frozen code-fit protocol diagnostics

These prepared files are not a change to an active run or an approved selection
protocol. They specify only steps and gum mode. Use the same explicit common
loss recipe in `../evaluation_fit_v1/`, model checkpoint, deterministic point
partitions, Adam learning rate, mesh resolution and sampling budget for each
single-factor comparison.

`scripts/compare_autonomy_evaluations.py --factor fit_steps` or
`--factor gum_code_mode` compares these controls with an identical checkpoint
SHA, common loss recipe, data/canonical/source/runtime and point partitions.
It permits only20->80 steps or temporary->fixed-mean gum, respectively, and
rejects another changed budget/policy field. Any failed case leaves the cohort
delta null. Training-factor comparisons require the same fit protocol in both
arms and cannot substitute these different fitting protocols for a training gain.

- `*_temporary20.json`: explicit reproduction of current 20-step temporary-code
  fitting, including gum. Omitting `--fit-protocol` retains existing numerics.
- `*_temporary80.json`: fresh fit from the same training means for 80 steps;
  change steps alone, do not resume or reuse another fit's codes.
- `*_fixed_gum20.json`: refit tooth codes for 20 steps while keeping gum at the
  exported original-present-trained mean (float64 mean used in model float32).
  Gum is excluded from Adam and the code-gradient requirement. Tooth means,
  loss, learning rate and point partitions keep their existing definitions.

`evaluate-cases` and `evaluate-cases-batch` accept `--fit-protocol <file>`.
Learned-train evaluation rejects it before loading a checkpoint. Batch evaluation
remains train/val only, reconstructs each checkpoint independently, checks file
SHA for drift and retains already completed results if a later check fails.
Single checkpoint provenance binds the file SHA and resolved protocol; result
budget includes the actual fit steps and gum mode. Fixed-gum artifacts are marked
`temporary_teeth_fixed_train_mean_gum`, not ordinary all-code fitting. Code
telemetry verifies their gum exactly against exported mean coordinates.

Final test remains separately sealed. If the chosen R4 protocol explicitly uses
one of these files, its existing complete-cohort final protocol must bind
`fit_protocol_sha256`, `fit_protocol_resolved_sha256`, and
`evaluation_budget_sha256` for the effective budget returned by `fit_budget`.
Supplying a new file without these bindings is rejected. These unit-test checks
use synthetic fixtures and do not authorize or execute actual test evaluation.

No standardized-q optimization or representation/bundle change is introduced.
The raw-code prior is retained. A post-fit gum swap is a different sensitivity
diagnosis from this independent fixed-gum tooth refit. Review all case/tooth
failures, held-out errors, area/topology and exact code artifacts before choosing
a subsequent R3/R4 protocol. Do not compare different protocols as an isolated
training-reduction effect.

Independent implementation validation:82 related CPU regressions passed
(137.379s unittest time;158.588s outer runner including startup/hash checks),
without skips, and default synthetic train/val output fields match frozen17f8
evaluation exactly (mesh stubbed). Initial30 related tests also passed; their
post-test wrapper failed on a string/Path conversion, retained the successful
raw log, and recovered status after verifying every input SHA without rerun.
These checks are development evidence, not target preflight or real model
quality. After the active job exits, a merged new commit requires a fresh target
preflight23+extra59 and real native/common/batch parity before diagnostic use.
