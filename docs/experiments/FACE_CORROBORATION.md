# Cached face corroboration diagnostic

Freeze this diagnostic before scoring. It changes no crop decisions, face
protection, model scores, training data or quality threshold. An unsupported
proposal is uncertain, not a non-face. Missing proposals remain unknown.

Use the exact real72 face packet and original25 local guard packet. Bind the
corresponding cached EfficientDet-Lite2 results by source ID and exact owned RGB
hash/dimensions. No new model inference or API calls. Use the existing weak Sol
head nominations only to evaluate real72 proposal coverage, separately by FIT/DEV.
Use the three visually reviewed original false-positive sources only as a
deliberately selected negative challenge, not corpus precision ground truth.

Group all full/tiled YuNet outputs as one family, UltraFace as one, and MediaPipe
BlazeFace/Face Landmarker as one conservative face-stack family. These groups
avoid counting repeated views and related pipelines as independent votes; no
statistical independence is asserted.

For each face proposal, test overlap with a different family at IoU0.1, plus
0.3/0.5 sensitivity controls. Person context means the proposal center lies
inside a cached Lite2 person box under the existing score0.3/max50 settings.
Compare all proposals, another-family support, person support, both and either.
Retain every proposal and its support signals, including uncertain proposals.

Count weak nominated centers covered by any retained proposal and whole weak
head boxes contained by any proposal. These are annotation-scope diagnostics,
not actual-face recall, identity association or complete head protection.
Do not select a runtime rule from reused DEV scores or add source-ID exceptions.

After inspecting the four bounded witnesses, register a separate secondary
association diagnostic: count nominated heads containing a proposed face
center. A face box can lie inside a broad head nomination without containing
that nomination's center, so the two directions measure different geometry.
Preserve the original primary point-coverage results and report both directions;
neither is actual-face recall. This adds no inference, gate or crop change.
