# Controlled R2 comparisons

R2 native validation fits new codes with the checkpoint's training recipe. The
prepared effective-reduction candidate changes both decoder training and that
fit objective. Native curves describe the combined recipe; they cannot isolate
decoder-training improvement. Keep those curves and their selected checkpoints.

Before ranking the baseline and effective candidate, evaluate the same declared
checkpoint epochs with the corresponding upper/lower `evaluation_fit_v1` recipe,
identical case IDs, fit/evaluation point partitions, steps, Adam rate, mesh grid
and mesh sample counts. Baseline native/common fitting must reproduce each other.
Record both training and code-fitting recipes and evaluation/checkpoint hashes.
The common objective uses all_points and retains the center denominator, weights
and epsilon of the baseline. No training decoder or embedding may be updated.

Use learned-train reconstructions to separate decoder limitations from held-out
code fitting. R1 currently has only two stratified learned-train diagnostic cases
per arch; that is insufficient for a train/val generalization conclusion. The
active R2 baseline will evaluate all 32 actual original trained cases per arch.
The tooth-only distance omits gum and unsupported generated caps, so inspect fixed
views, field boundaries, component areas and normal/semantic errors alongside it.
Many disconnected teeth can be legitimate; inspect component areas/labels rather
than requiring one connected component or treating every disconnected island as
an anatomical error.

If common-objective comparisons indicate an improvement, quantify uncertainty
from mesh sampling seeds and resolution on a fixed representative train/val
subset, and repeat material training improvements. Increasing fit steps or grid
resolution is a separate inference-budget diagnostic; all compared candidates
must then receive the same revised budget. Do not rank unequal point partitions,
resolution or fit budgets, drop failed cases, move the isovalue or relax boundary
checks. Valid geometry for all selected cases remains necessary, not sufficient.
Freeze the resulting quality protocol before R3/R4; no clinical threshold is
claimed by the coarse diagnostic protocol.

Merge these evaluation changes only after the frozen 4fd0d5a driver completes and
its results are inspected. Re-run target preflight on the next committed version.
Run a small real-dental native/common parity gate before another bounded GPU
comparison; passing synthetic regression alone does not replace that gate.
