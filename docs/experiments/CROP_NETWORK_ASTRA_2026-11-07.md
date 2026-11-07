# Astra crop-network architecture consultation, 2026-11-07

User-requested GPT-6 Astra (high) read-only consultation on source f2465355.
This is design advice, not a measured or trained model. The coordinator preserves
the recommendation here; no proposed speed, memory or quality gain is established.

**I recommend a shared convolutional encoder with multiscale region pooling, a small candidate-ranking head, and deterministic constrained search.** Keep direct box prediction as a later proposal generator.

I inspected clean revision `f2465355`, the requested sources, and the existing runtime kernels. No files changed, builds ran, or agents launched. Crop-student quality, latency and memory remain unmeasured; the recommendations below are proposed experiments.

1. **Encoder: retain w24 initially, but expose finer features.**

   Keep the stem, inverted residual stages and stride-16 bottleneck; remove the saliency decoder. Add a small channel projection from the stride-8 `enc2` output alongside projected stride-16 features. Pool these separately and concatenate their descriptors—no feature-pyramid decoder is necessary.

   This preserves finer boundary information while retaining semantic context. Fractional coverage weighting in the [current head](/home/lilith/work/zen/zensally/training/crop_ranker.py:29) prevents empty pooling, but cannot recover detail lost during resizing and downsampling.

   For variable source aspects, use a bounded set of aspect-preserving rectangular input shapes, dimensions divisible by 16, with comparable pixel budgets. Select the shape from source geometry, independently of requested crops. Export fixed plans and bound scratch by the largest supported shape.

   Keep square stretch and letterbox as controls. Existing saliency results favor stretch, but do not establish its superiority for a newly trained composition model.

   Local views are worthwhile only when held-out results show that detail loss limits composition. Prefer cached, source-dependent overlapping views reusable across requests. Candidate-local encoding should be a separately budgeted finalist-refinement option.

2. **Highest-value change: replace isolated samples with pooled regions.**

   Nine interior samples can overlook a subject between samples; exterior midpoints can overlook the object being discarded. On narrow crops, several samples also become nearly redundant. Large receptive fields further blur the distinction between “inside” and “outside.”

   Replace them with:

   - Spatially ordered 3×3 **area-pooled interior bins**.
   - Four pooled, disjoint discarded bands, each with area and presence indicators.
   - Narrow inside/outside boundary strips from stride-8 features.
   - Cached global context.

   Project channels before pooling. Start with directly accumulated fractional-cell pooling; compare summed-area tables only if candidate volume justifies their construction and retained memory. Define empty-band behavior explicitly.

   Condition the MLP on normalized coordinates, retained area, source/target log-aspect, mode and face context. Add a separate acceptability output: relative ranking alone cannot identify a pool in which every crop is bad.

   Retained/discarded-region modeling has precedent in [GAIC](https://openaccess.thecvf.com/content_CVPR_2019/html/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.html); this particular multiscale head remains a hypothesis.

3. **Search: preserve multiple good answers without making the grid authoritative.**

   Minimal fixes maximum feasible area and searches placement; unchanged aspect bypasses learned cropping. Maximal searches placement and scale, always including the broad maximum-area candidate. Never add a reward for removing more area.

   Start with center, interior, endpoint and face-feasible placements. Refine several spatially distinct high-ranking candidates through bounded position/scale neighborhoods, retaining exact geometry constraints. Measure candidate-set coverage separately from ranking quality: a perfect ranker cannot select an absent composition.

   | Alternative | Assessment |
   |---|---|
   | Candidate ranking | Recommended first: inspectable proposals, explicit constraints, reusable image analysis. |
   | Direct multimodal boxes | Useful later for proposals; needs diversity/matching supervision and can miss valid alternatives. Avoid averaging coordinates across incompatible good crops. |
   | Factored boundary utilities | Useful for pruning or auxiliary features; independent edge scores cannot fully represent relationships between subjects, negative space and opposing boundaries. |

4. **Runtime: reuse convolution machinery; add pooling and planning.**

   Existing operators cover the dense stem, pointwise projections, strided/dilated depthwise convolution, ReLU, residual addition, squeeze/excitation means, sigmoid and channel multiplication. Fold batch normalization offline.

   Reuse `zensally-micro`’s pointwise/depthwise implementations and zenjpegai’s packed tensors. Dense head layers can use pointwise convolution over candidate descriptors. No deployed transformer, gradient engine, general ONNX optimizer or transposed convolution is needed.

   The actual additions are fractional region pooling, descriptor assembly, candidate search, and fixed-plan export for the selected shapes. The existing resize helper is **not** a substitute for arbitrary `grid_sample`: its sampling contract differs.

   Retain projected maps/global context once; process candidates in bounded chunks with reusable descriptor storage. Avoid materializing every candidate’s sampled tensor. Allocate checked capacities fallibly at context construction, accept strided pixels, and keep CPU dispatch through archmage/magetypes.

   The current [workspace](/home/lilith/work/zen/zensally/crates/zensally-micro/src/workspace.rs:12) provides reuse machinery, but some shared kernels allocate their outputs. Fixed context-owned scratch therefore needs explicit output-buffer plumbing and allocation verification in both std modes.

5. **Training: learn preferences and rejection separately.**

   Start with the existing AdamW/cosine setup and within-task Bradley–Terry loss. Warm up the new head, then fine-tune the encoder; compare training from scratch. Treat saliency initialization as an ablation, not composition ground truth.

   Reviewed ties can use a symmetric preference target; uncertainty supplies no invented ordering. “Reject all” trains acceptability, not arbitrary pairwise winners. Calibrate abstention using human judgments: raw pairwise scores have no reliable absolute threshold.

   VLM coordinates are proposals only. Project them into legal geometry, render the resulting pixels, and judge those actual crops. Preserve raw and projected coordinates. Randomize presentation, test swapped order, permit multiple acceptable answers, and audit teacher disagreement against humans.

   Broaden proposal sources and mine missed alternatives on training sources. Validate on unfamiliar proposal distributions to expose candidate-set bias.

   Freeze source-family roles before labeling; derivatives inherit their source role. Quarantine cross-role relatives and audit near duplicates. Existing validation galleries stay excluded. Real extreme-aspect photographs must complement synthetic stress cases. Smaller teacher previews require a preference-stability audit; small faces and boundary cuts still need detailed review.

6. **Face safety belongs entirely to geometry.**

   Keep every supplied face protected regardless of confidence, size or learned importance. Transform bounds through the same orientation mapping as pixels, then conservatively rasterize them outward.

   For reduced integer target ratio \(p:q\), exact integer crops have dimensions \(kp\times kq\). Choose permitted sizes, derive integer origin intervals containing the protected union, and reject empty intervals. Recheck bounds, aspect and every protected face after final rasterization and against the renderer’s actual retained footprint.

   Unknown coverage, infeasible geometry, rejected candidates or nonfinite scores return contain/pad; unchanged Minimal remains unchanged. Tiny faces, separated groups and opposite-edge faces need explicit cases.

   A caller’s reviewed-coverage assertion is a trust boundary. Neither empty detections nor learned confidence establishes universal face coverage.

7. **Choose the design through staged, frozen gates.**

   First compare point sampling against pooled regions with the encoder/search fixed. Then isolate stride-8 detail, exterior/boundary features, aspect-shaped inputs, search refinement and optional local views.

   Require zero protected-face violations and exact Minimal invariants. Measure human composition preference with source-family uncertainty, acceptable-candidate coverage, reject-all detection, unnecessary padding, removed area and failures by source/target aspect—including 1:8 and 8:1.

   Benchmark preprocessing, analysis, first query and additional queries separately; measure candidate-count scaling, cold/warm behavior, allocations, peak RSS and cache behavior. Verify export/operator parity and selected rectangles across CPU tiers. Existing saliency-runtime measurements and tolerances do not establish crop-ranker performance or authorize new approximation.
