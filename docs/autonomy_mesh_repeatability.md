# Fixed-mesh sampling repeatability

`python scripts/resample_autonomy_meshes.py --evaluation runs/ACTUAL_EVALUATION --output runs/NEW_REPEATABILITY`
uses saved train/val meshes and evaluated scan points on CPU. It first reproduces
the original per-tooth forward/reverse means and P95 values within 1e-8 mm, using
the original mesh-sample count and seed. Missing present tooth labels cause a
failure; failed original cases retain a null requested-cohort score. Test is
rejected through provenance before opening its metric data.

Defaults are 1024 and 4096 area-weighted mesh samples per present tooth, with five
mesh-only seeds 20261006–20261010. The observed points, learned/fitted codes,
Marching Cubes geometry and face labels stay fixed. This measures random mesh
sample variation and count sensitivity, not code-fitting, point-partition,
training-seed, patient or grid-resolution uncertainty. It must not be emulated by
changing the validation seed, which would also change the fit/evaluation points.

Case/tooth tables, original input hashes, analysis source hashes and CPU runtime
versions are retained. Empirical min/max and population standard deviation across
five mesh seeds are descriptive; they are not a confidence interval or a clinical
acceptance threshold. Do not rank candidates with different protocols or declare
R3/R4 readiness from this repeatability result alone.

The first probe uses all four completed R1 diagnostic evaluation directories
inside the active R2 run: two learned-train and four val cases per arch, already
checked against the original native evaluator. This is a 12-case diagnostic
sampling study, not a representative validation population. Saved surfaces can
be analyzed without another GPU job or changes to the active training checkout.
