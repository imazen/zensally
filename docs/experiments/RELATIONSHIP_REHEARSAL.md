# Old-quality rehearsal with relationship students

Register12 fits before training: old replay only, replay+new contrastive loss,
replay+contrastive+real relationship features, and matched donor features;
seeds31/137/911. Freeze image encoder, first scorer layer and original scalar
base. Train the same96->32 ReLU->32 residual student in every arm.

Use the verified original94FIT/25DEV hidden/target cache and new48FIT/24DEV
cache. Preserve roles/families, candidates, old objective, all unknown labels
and three-grid tolerance. Reject cross-role family overlap. No TEST, new API
calls or teacher inference. FIT-only input normalization uses new eligible
candidates; real/permuted targets share matched-seed teacher FIT normalization.

Run88 epochs plus zero, batch8 old sources plus8 cyclic new sources, AdamW
lr0.001/decay0.0001/clip5 FP32/two CPU threads/4GiB cap. Old rank loss uses
the original function and family weights; source hash phase cycles one query
per epoch. Add0.25 new explicit-positive/negative contrastive loss when enabled
and0.1 source-balanced eligible-candidate feature MSE in the feature arms.
Feature MSE uses all22 queries; unknown candidates do not enter contrastive loss.

Select only epochs whose same-platform old essential regret does not exceed
epoch zero. Then minimize new essential family head cuts, maximize reference
matching, minimize old regret and choose earliest tie. Include epoch zero.
Record the CPU baseline separately and require unchanged original full-model
CUDA quality evaluation of selected heads; CPU selection does not replace the
shipping quality gate. Retain every epoch and initial/selected/terminal model,
optimizer/RNG, scores/choices and normalization states. No model promotion
without numerical/runtime and artistic validation.

All12 selected fits subsequently pass the unchanged original full-model old
quality gate. The [measured controls](../../benchmarks/relationship_rehearsal_2026-11-09.md)
find a nominated-head-cut improvement from new contrastive labels, with no
incremental benefit established for real relationship feature targets. Keep
the weak-label and artistic/runtime limitations; no model is promoted.

The subsequent [native head study](../../benchmarks/residual_native_runtime_2026-11-09.md)
measures embedded zenpredict weights, exact fixed-panel crop-choice agreement,
head-only paired costs and sudo perf/heaptrack evidence. Full encoder/pooling
cost and artistic suitability remain outside that measurement.

The later [complete-neural study](../../benchmarks/rehearsal_network_2026-11-09.md)
qualifies both raw and gather encoder/query paths on191 FIT/DEV sources with
zero candidate ORT crop-choice changes in4202 requests. Reusing the existing
fixed gather lowering reduces measured reference latency while retaining all
weights. Full native deployment, image preprocessing, safety guards and
artistic crop quality remain outside those timing claims.

The subsequent [actual-pixel review](../../benchmarks/rehearsal_witnesses_2026-11-09.md)
retains five deliberately selected DEV witnesses, including both weak-label
regressions. Full neural inference reproduces every displayed model choice.
Visible head/body failures keep promotion closed despite the old-quality pass.

The [local face candidate audit](../../benchmarks/rehearsal_face_admissibility_2026-11-09.md)
then tests existing detectors without changing scores or candidates. Local
whole-face geometry reduces some weak cuts while retaining missed-head and body
framing failures. Containment and conditional failure rates remain separate;
neither model nor filter is promoted.
