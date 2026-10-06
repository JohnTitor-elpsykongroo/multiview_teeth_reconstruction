# Autonomous checkpoint evaluation

`evaluate-cases` complements the original `evaluate-checkpoint` validation command.
It requires an explicit split. The original command continues to accept val only.

- `train`: reconstruct original cases in the checkpoint's actual trained subset
  with their learned embedding rows. No code fitting. Evaluation points are a
  deterministic resample of the training scan; they are not claimed to be unseen
  during resampled training. Mirrored and untrained rows are rejected.
- `val`: initialize temporary codes from original present train rows and fit them
  with a frozen decoder. Defaults to the configured original val cohort.
- `test`: same inference procedure, on all original test rows. Requires an
  explicitly frozen final protocol. No test row relabeling or subset evaluation.

The CLI's `--case-id` can be repeated for train/val. It does not change the
checkpoint's training subset or permit test rows in a val evaluation. All runs
use a fresh output directory and retain failures in the requested denominator.
The selection score is null if any requested case fails. Successful-only
descriptive statistics are labeled separately and are never selection scores.

```bash
python third_party/DMM/dmm_cli.py evaluate-cases \
  --config configs/autonomy/r2_baseline/upper.json \
  --checkpoint runs/ACTUAL_RUN/final.pth --split train \
  --case-id 00OMSZGW_upper --device cuda --output runs/NEW_TRAIN_EVALUATION

python scripts/preview_training_evaluation.py \
  --evaluation runs/NEW_TRAIN_EVALUATION --output runs/NEW_TRAIN_PREVIEWS
```

Use an actual checkpoint and a compatible config. The config's resource SHA,
specs, canonical reference, actual training subset and recipe are checked before
evaluation. Decoder compatibility is checked independently from evaluator
provenance so additive evaluation tools can inspect earlier compatible weights.
Export still uses the exact training source; this feature does not waive export
or resume signatures.

The final test protocol JSON must contain:

```json
{
  "stage": "R4_FINAL_FROZEN",
  "checkpoint_sha256": "actual checkpoint SHA256",
  "manifest_sha256": "actual frozen manifest SHA256",
  "evaluation_budget_sha256": "budget_digest(resolved_config.validation)",
  "case_ids": ["all original test case IDs in lexicographic order"]
}
```

The researcher creates this only after selecting the final checkpoint on val and
freezing the protocol. Pass it with `--split test --test-protocol PATH`, without
`--case-id`. The file binds the intended experiment; it does not prove that a
researcher has never inspected test before. A later investigation must honestly
record any prior test use. This project does not enforce a destructive global
one-shot lock: interrupted evaluations retain partial records and can be
reproduced under the same frozen protocol, without choosing new hyperparameters.

Each case records mixed and component field/gradient/semantic diagnostics,
point-index digests, elapsed time and geometry errors. `case_NNNN/field.json`
records volume extrema, signs and domain boundary counts even when a surface is
invalid. Valid-boundary zero meshes are saved as `mesh.npz` before checking
whether every present tooth has a blended face. An incomplete tooth mesh is a
failure diagnostic, not an accepted reconstruction. No shifted isovalue or
relaxed domain boundary is used.

The previews show fixed LPS front, side and occlusal axes at a shared scale.
Face colors show dominant FDI, gum is gray, black marks the evaluated scan
samples. Pillow painter ordering is a diagnostic view, not the soft-semantic
differentiable rendering used for image fitting. Only area-sampled surface
distances are reported; no exact point-to-triangle or clinical claim is made.

Regression coverage includes trained-row selection, wrong split/mirror
rejection, train learned-code use without Adam, held-out fitting with exact
decoder/embedding state preservation, disjoint deterministic point partitions,
failure accounting and retained no-surface diagnostics, and test protocol/hash
and full-cohort binding. Existing training, resume, export and val tests remain
required. Real dental small-case evaluation is still required before long runs.

## Common code-fitting objective for controlled comparisons

Default val/test fitting uses the checkpoint's training recipe. Consequently a
training loss reduction change also changes the temporary-code inference loss.
That comparison describes a combined training/inference recipe, and cannot
attribute improvement solely to the trained decoder.

`evaluate-cases --fit-recipe configs/autonomy/evaluation_fit_v1/upper.json`
uses a separate, fully resolved loss recipe only for temporary val/test codes.
The decoder and training embeddings stay frozen; the system's training recipe
is restored even on a case failure. All step counts, Adam rate, sampling and
geometry budgets still come from the original validation protocol. Train
reconstruction rejects this option because it uses learned codes without fitting.
The upper/lower `evaluation_fit_v1` files freeze the R2 baseline recipe, including
all weights, reduction, center denominator and normal epsilon.

Use the same common recipe for both baseline and controlled candidate val
evaluations, and retain native-recipe diagnostics separately. Baseline native and
common fits must reproduce each other before comparison. Do not overwrite native
`best_val` evidence or retroactively relabel it as a common-objective selection.
Compare checkpoint curves under the declared common objective when making the
next-stage recipe/checkpoint decision. The active 4fd0d5a baseline is immutable;
this capability requires a future merge and fresh preflight before GPU use.

For test with an explicit recipe, the final frozen protocol must additionally
bind `fit_recipe_sha256` (the exact JSON file) and `fit_recipe_resolved_sha256`
(`budget_digest(load_fit_recipe(path, arch))`). Without an override, the fit loss
is already bound by the selected checkpoint; a protocol that names an override
cannot silently fall back to that default. Provenance records the training and
fitting recipes, file SHA, manifest SHA and canonical identity separately.

## Bounded checkpoint batches

`evaluate-cases-batch --checkpoint PATH ...` evaluates independent checkpoints of
one frozen train/val config and common fitting recipe. It fully validates the
dataset once per batch, then rebuilds the model, embedding table and temporary
codes separately for each checkpoint. Every evaluated case still rehashes its NPZ.
Config/source/fit-recipe/checkpoint hashes and manifest/canonical/spec references
are checked for drift; per-checkpoint provenance and partial failures are retained
under `checkpoint_0000`, etc., with a `batch.json` ledger. Geometry failures preserve
null cohort scores. Process/integrity failures stop the batch with completed results
retained. Test is excluded from this research batching command.

Repeated full manifest validation took about six minutes per individual R2
evaluation, while four coarse val cases took about 77 seconds. Batching the four
matching epochs avoids repeating that full validation four times for immutable
inputs; it does not remove case hashing, increase fit/grid budgets or reuse fitted
codes. Synthetic regression must match individual evaluation exactly except wall
time, and the next GPU run must pass a real small batch/individual parity gate.
