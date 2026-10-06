# R2 effective early failure diagnosis

Observation recorded 2026-10-06 18:08 UTC. Active GPU control remains frozen at
`17f8c07c3a1a582b3916a5fff6e6d74b855bd983`; CPU development was frozen at
`da5cd3f5c6d5f9c737aef3b3a30edcfbd78a8d0a`. Neither the training checkout nor
the frozen evaluation protocol was changed. This is a partial diagnostic, not
completion of the controlled comparison or approval of R3/R4.

## Native upper validation

The four fixed cases are `01343APK_upper`, `0140W3ND_upper`,
`0145NKWZ_upper`, and `014WKP6A_upper`. All failed at every saved epoch:

| Epoch | Valid cases | Failure |
| --- | --- | --- |
| 5 | 0/4 | no finite zero surface |
| 10 | 0/4 | missing blended surface component 11 |
| 15 | 0/4 | missing blended surface component 11 |
| 20 | 0/4 | first three missing 12; last missing 13 |

Scores remain null. No missing tooth was omitted, iso-level shifted, boundary
condition relaxed, or failed case removed. The bounded driver completed upper
training and entered lower training. Its later common-fit evaluation remains
necessary because native reduction changes the temporary fitting objective too.

## Fixed trained-point response

CPU experiment `runs/autonomy_setup/latent_R2_effective_upper_20261006T1800`
binds baseline/effective checkpoints at epochs 5 and 10, exact config and frozen
manifest/canonical SHA, and two original learned-train observation artifacts from
`runs/r2_baseline_20261006T134608Z/upper_r1_train`. Cases were predetermined:
`01328DDN_upper` and `01422MSK_upper`, first 16 saved points per present component.
Original-present-trained component means exclude mirrors and absent rows.

| Epoch | Recipe | Learned-code semantic error, case 1 / case 2 |
| --- | --- | --- |
| 5 | baseline | .062500 / .109375 |
| 5 | effective | .912500 / .895833 |
| 10 | baseline | .070833 / .062500 |
| 10 | effective | .850000 / .776042 |

At epoch 10, absolute surface SDF means are .017970/.021798 model units for
baseline and .043113/.042375 for effective. These are field residuals, **not mesh
distances**. Effective train-mean replacement still has semantic errors
.870833/.791667. Zero-code replacement also retains high errors. The experiment
does not prove latent collapse or identify the cause of the training failure.

The CPU computation completed in 50.792 seconds within a 300-second, eight-thread
budget. A subsequent review wrapper used a wrong JSON key and failed; its status
and raw study were retained. `review_effective_upper_probe.py` rechecked every
bound SHA and recovered the review without repeating the numerical experiment.

## Fixed trained-code mesh check

Experiment `runs/autonomy_setup/grid_R2_effective_train_20261006T1810` uses the same
two frozen original-train observation sets, checkpoint-own learned codes, grid48,
seed20261006, and 1024 mesh samples/component. There is no temporary code fitting.
Baseline and effective are evaluated at the same epoch 10. Effective training
means form a separate sensitivity diagnostic, not a substituted evaluation.

| Recipe / code | Case 1 | Case 2 |
| --- | --- | --- |
| baseline learned | valid, 1.512915 mm | valid, 1.880516 mm |
| effective learned | missing tooth surface, null | missing tooth surface, null |
| effective train mean | missing tooth surface, null | missing tooth surface, null |

Every effective generated mesh face has dominant label 0 (gum); all present
tooth labels are missing, not merely the first reported FDI11. Baseline contains
every present label. Baseline gum area shares .859342/.878313 are diagnostic;
they are not an anatomical quality threshold. Failed effective exact codes,
field summaries and meshes are preserved and SHA-bound. The study SHA is
`77a6cdb7054b49a35d29bd3a040a3123c6c9b96775ef813ef5af7d624c72de9b`.

Six predetermined checks completed in 55.660 seconds within a 600-second,
eight-thread CPU budget. Static arch previews use the existing inferior view for
upper. The first Windows review could not resolve WSL paths; that failure was
retained and the exact review script ran in WSL without repeating geometry.
All CPU processes explicitly disabled CUDA. Existing geometry and artifact
contracts were used; no environment changes or second GPU job occurred.

## Interpretation and next decision

Effective's failure occurs with its own trained codes and cannot be attributed
solely to validation temporary code fitting. On these frozen examples, the
blended geometry is dominated by gum and lacks tooth surfaces.

Changing the reduction multiplies component own-point surface, normal and
template-normal losses by `total_points/own_points` at identical outputs. With
15 components, 256 points/component and 2048 off-surface points, this factor is
23. Blend BCE, center, latent, Eikonal, correction and smoothness retain their
definitions. Global terms change by different factors, and the flip denominator
depends on current geometry. Thus loss balance and gradient clipping warrant
diagnosis. These observations do not establish a causal mechanism, and clipping
fraction alone is not a stop or acceptance rule.

Retain all checkpoints and native failures. Complete the active lower run and
common matched-epoch control within its frozen budget. Do not promote effective
on loss reduction or a partial valid subset. If common geometry fails, reject
this uncalibrated recipe and retain baseline as the reference; any repair must be
an explicitly frozen single-factor experiment. The latent1e5 option remains
unrun and is not justified as a remedy for missing surfaces by these results.
Baseline still requires fitting/deployment protocol diagnostics and tail/area
review before R3. No test data were read or scored.
