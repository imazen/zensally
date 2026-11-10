# Quality-gated old replay and relationship students

All12 selected teacher-free students pass the unchanged original full-model
old quality gate. Old FIT rehearsal resolves the earlier enabled-student
regression. New contrastive crop labels improve nominated-head cuts versus
replay alone, while real relationship feature supervision adds no measured
advantage over the matched contrastive or permuted controls. No model is
promoted and artistic composition remains unproven.

| Three-seed essential DEV family means | Head cuts | Reference matching | Original old gate |
|---|---:|---:|---:|
| Replay only |0.402428|0.110483|3/3 pass|
| Replay + contrastive crops |0.364355|0.121579|3/3 pass|
| Replay + contrastive + real features |0.365096|0.117922|3/3 pass|
| Replay + contrastive + permuted features |0.360815|0.120738|3/3 pass|

Contrastive versus replay-only cut difference is-0.038073, paired family
95-percentile interval[-0.065930,-0.013724]. Each seed improves. Matching
difference0.011096 has interval[-0.000607,0.022426]. Real features versus
contrastive-only cut difference0.000741 spans[-0.018924,0.021619]; matching
difference-0.003657 spans[-0.014628,0.004902]. Real versus permuted feature
differences also span zero. Use2000 family resamples, seed31, with seeds
averaged before resampling,17 families/139 essential DEV requests. All are
selected-DEV weak-label diagnostics, not independent significance or human gold.

The original old full-model C002 essential regret reproduces
0.24700142443180084 on25 DEV photos/21 families/448 valid essential queries.
Selected contrastive-only seeds31/137/911 score0.2445547878742218,
0.24497230350971222 and0.23651084303855896. Real-feature seeds score
0.2454291582107544,0.24481792747974396,0.24108439683914185. Every control also
passes. Replay-only seed137 retains epoch zero; all remaining selected
checkpoints are trained epochs, retained explicitly in the analysis.

CPU selection uses its own epoch-zero old regret0.24700143933296204. This
platform reduction difference does not relax or replace the original gate:
every selected enabled model is independently evaluated through the unchanged
original CUDA evaluator, with the fixed quality source files byte-checked.
All12 pass that original bound. Original roles/families/candidates, unknown
labels, three-grid numerical tolerance and TEST closure remain unchanged.

Training uses the registered frozen base plus96->32 ReLU->32 feature probe,
old94 FIT ranking objective and new48 FIT acceptable-set/feature terms. Original
rank objective/family weights are reused; no old DEV targets enter gradients.
New contrastive terms see only accepted positives and explicit face-cut
negatives. Old source/query phases and cyclic new batches are fixed; feature
MSE uses all eligible candidate queries. AdamW lr0.001/decay0.0001/clip5,
88 epochs plus zero,8 old plus8 new sources per step, FP32/two CPU threads.
Old-regret nonregression gates selection, followed by new family head cuts,
reference match, old regret and earliest epoch. All initial/selected/terminal
model/optimizer/RNG/scores/choices and histories remain retained.

Nonlinear arithmetic compaction changes zero eligible choices and zero crop
edge pixels for each of12 models'1,584 new-panel requests; maximum absolute
score difference1.1920928955078125e-6. Tests independently exercise family/query
balance, candidate masks, empty-source rejection, frozen base and compact
arithmetic. This validates arithmetic on the cache, not Rust export or runtime.

Training takes80s/run-heavy peak RSS1.32GiB, min available21876MiB under4GiB/
two-thread cap. Original quality evaluation takes3s/1.55GiB under4GiB;
arithmetic2s/0.78GiB and analysis1s/0.59GiB under2GiB. No throughput saturation
or serving-cost claim follows. No API calls or additional teacher inference.

Evidence roots under `/mnt/v/output/zensally/`: `relationship-rehearsal`,
`relationship-rehearsal-oldgate`, `relationship-rehearsal-arithmetic` and
`relationship-rehearsal-analysis`, all dated2026-11-09. Reproduce with
`just composition-fit-relationship-rehearsal`, then
`just composition-evaluate-attention-fallback ... --nonlinear-students --rehearsal`,
`just composition-audit-attention-fusion` and
`just composition-analyze-relationship-rehearsal`.
The [registered protocol](../docs/experiments/RELATIONSHIP_REHEARSAL.md) remains
separate. Actual artistic evaluation and complete serving-cost measurements
are still required; the remaining36% weak head-cut rate is not acceptable as
a face-safety guarantee.
