# Attention controls and original-quality fallback

The15 trainable-base attention heads all fail the original old-cohort essential
regret gate when attention is explicitly disabled. The original C002 baseline
reproduces0.24700142443180084 on25 DEV photos/21 families/448 valid essential
queries. No threshold changes and no promotion follow.

| Fallback arm | Three selected-seed essential regrets |
|---|---|
| Retrained base |0.277793,0.272451,0.275076|
| Pose |0.287039,0.291239,0.278331|
| Pose + gaze |0.346228,0.353119,0.357275|
| Pose + gaze + paths |0.338313,0.338082,0.339979|
| Permuted full |0.293846,0.285777,0.286479|

This fallback disables the entire normalized attention contribution, keeping the
trained base weights. It is not equivalent to supplying zero raw features and
is not student distillation. Full-head evaluation with old-source attention
features remains unmeasured. The failure shows that simply disabling teachers
does not preserve the original scorer after unconstrained base fitting.

Direct matched comparisons on the new139 essential DEV queries/17 families
also temper the earlier suitability result. Pose+gaze versus permuted-full
reference matching improves0.041338, with family-bootstrap95-percentile
interval[-0.002334,0.092437]. Full real features versus permuted-full improve
0.041151,[0.008340,0.080597]. Full features versus pose+gaze give-0.000187,
[-0.025210,0.024837]. All corresponding head-cut difference intervals include
zero. Seed deltas, selected epochs and all five comparisons are retained.
These are selected-DEV weak-label diagnostics, not independent significance,
human artistic gold or face safety.

Fallback evaluation takes2s/run-heavy peak RSS1.55GiB under4GiB/two threads,
CUDA process fraction0.2. Direct comparison takes1s/0.59GiB under2GiB.
They are validation jobs, not runtime benchmarks. Frozen quality source files
are byte-checked against9f126924 before the original evaluator is used.
Evidence: `/mnt/v/output/zensally/attention-fallback-gate/2026-11-09/` and
`/mnt/v/output/zensally/attention-controls/2026-11-09/`.

The separately registered residual-only panel freezes the original base weights
and trains only attention weights, retaining the original panel and all gates.

All15 residual-only fits complete. Original base weights are asserted unchanged
after every optimizer step. All15 fallback evaluations reproduce the complete
original old-cohort prediction records exactly, not just its summary regret.

| Residual-only arm | Essential head cuts | Reference matching |
|---|---:|---:|
| Frozen original baseline |0.408395|0.111137|
| Pose |0.371980|0.118139|
| Pose + gaze |0.388787|0.111137|
| Pose + gaze + paths |0.363908|0.141529|
| Permuted full |0.391614|0.102453|

Full residual features versus permuted improve matching0.039076, with paired
family95-percentile interval[-0.004202,0.088515]; head-cut difference-0.027706,
[-0.113448,0.056231]. Both span zero. Residual fitting takes21s/peak RSS1.05GiB,
min available22322MiB,4GiB/two threads. Its old fallback check takes2s/1.55GiB.
Evidence roots are `attention-residual-fits`, `attention-residual-fallback` and
`attention-residual-controls`, each under the2026-11-09 output directory.
No feature-enabled old-cohort quality, independent artistic judgment, new
runtime measurement or teacher-free student transfer is established here.
Retain both panels and preserve all promotion requirements.
