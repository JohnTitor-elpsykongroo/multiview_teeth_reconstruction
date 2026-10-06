# Optimizer diagnostics

Training now saves `optimizer_steps.jsonl` with original case ID, epoch, scalar
loss, pre-clip global/group gradient norms, the actual global clipping coefficient
and elapsed step time. Epoch progress adds norm/coefficient min, median, P95 and
max, clipped-step fraction, training time and time including validation before
checkpoint writes. GPU scalar transfers already synchronize preceding work;
these timings include data loading/checks and logging, not GPU kernel profiling.

Clipping stays global over all system parameters. The coefficient uses PyTorch's
`max_norm / (total_norm + 1e-6)`, capped at one. Group norms precede clipping.
The coefficient is a gradient diagnostic, not an Adam parameter update magnitude.
No RNG draw, weight change, loss change or extra backward pass is introduced.
Persistent clipping calls for investigation of losses and gradients; clipping
alone is not a model quality or automatic stopping criterion.

Regression tests cover the clipped/unclipped denominator and exact weights,
Adam state and RNG preservation with scalar summary enabled versus disabled.
An additional two-epoch CPU comparison with the original `fecef56` trainer at
clip threshold 0.5 produced exactly equal weights, Adam, scheduler, RNG and loss;
its retained report is `runs/autonomy_setup/instrumentation_parity.json` in the
development checkout and will be copied into the next experiment's provenance.
Real host preflight and dental evaluation remain required before R2 training.
