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
