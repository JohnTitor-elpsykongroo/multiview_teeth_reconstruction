# Train/val evidence summaries

`python scripts/summarize_autonomy_run.py --run runs/ACTUAL_RUN --output runs/NEW_SUMMARY`
reads completed evaluation directories and snapshots complete training JSONL
records. It works on CPU without loading checkpoint weights or original scans.
Output includes every requested case in `cases.csv`, tooth geometry and component
field diagnostics in `teeth.csv`, all original/raw/weighted terms with their
case denominators and clipping distributions in `training_curves.csv`, and a
Markdown/JSON summary with input and script hashes. Model-unit SDF and mm surface
distances remain separate; reverse distance includes unsupported generated caps.
When a saved mesh exists, case tables also report gum area fraction, area per FDI,
connected-component areas, largest-component area fraction and generated extent.
These expose geometry excluded by the tooth-only score. Multiple disconnected
teeth and unsupported caps are not automatically treated as anatomical errors;
these fields have no automatic rejection threshold.
The FDI 0 row is the gum component; its field diagnostics have no tooth distance
score. Mesh and field artifact hashes are included in the snapshot's inputs.

The summary checks failure and requested denominators, null selection scores for
failed cohorts, equal-tooth case scores and equal-case cohort scores. It rejects
test in provenance before opening metric content. Incomplete evaluation folders
are not scored; a partially appended training JSONL line is excluded from the
snapshot. Existing output directories are rejected. This tool reports evidence;
it does not select a model or assert quality acceptance, and evaluations with
different protocols are shown separately rather than ranked.

An active run's summary belongs in the independent development checkout. Do not
merge code or alter the training checkout until its driver exits. The next stage
can use committed analysis tools after a fresh target proof.
