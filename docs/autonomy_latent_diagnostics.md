# Original latent diagnostics

`scripts/inspect_autonomy_latents.py` reads frozen checkpoints on CPU. Per-component
statistics include only explicitly present original cases in `training_case_ids`.
Absent, mirrored, untrained and held-out rows cannot enter means or covariance.
Statistics use raw covariance without the inference bundle's covariance floor;
participation ratio describes variance distribution, not representation quality.

Optional probes reuse observed points saved by a successful learned-train evaluation.
The tool checks train provenance before reading metrics, original trained membership,
presence, units, manifest/canonical binding and decoder compatibility. It freezes
the decoder and compares the learned codes with original-present component means
and zero codes, on a fixed quota of points per component. It neither fits codes nor
generates meshes. Original artifact hashes and selected point hashes are recorded.

Example (paths refer to immutable completed files):

```sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scripts/inspect_autonomy_latents.py \
  --checkpoint runs/R2/upper_baseline/epoch_000005.pth \
  --checkpoint runs/R2/upper_baseline/epoch_000015.pth \
  --probe-evaluation runs/R2/upper_r1_train \
  --case-id 01328DDN_upper --case-id 01422MSK_upper \
  --output runs/analysis/latent_response.json
```

During an active training job run the tool from the independent checkout and write
there; never mutate its training checkout or use CUDA. Checkpoint loading is
CPU-only and sequential, with memory mapping to avoid copying optimizer tensors.
Use uniquely saved epoch checkpoints rather than mutable best checkpoint aliases.

Small code magnitudes alone do not demonstrate collapse: a hypernetwork can amplify
small differences. SDF deltas are field diagnostics in model units, not geometric
millimeter errors. Warped-point displacement is a transformation diagnostic in mm,
not observed-to-mesh distance. Mean/zero code substitution is not an unseen-patient
experiment. These probes do not approve a model, justify changing regularization
alone, or measure fitting, grid, patient or training-seed uncertainty.
