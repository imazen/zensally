# Native query experiment

Research-only FocalNet candidate scoring from an existing 32×32×17 feature
cache. This binary does not encode an image, detect faces or authorize a crop.
The large encoder remains separate. No runtime or model is promoted.

It uses f64 summed-area region means, ordered f32 geometry, a fixed 62-value
LayerNorm array and the existing zenpredict dense runtime. `ordered` and
`algebraic` normalization are separate experiments. Rust 1.98 algebraic methods
are used explicitly; the measured compiler is recorded with each build.
All input sizes are bounded; input-driven vector reservations are fallible.

`build.rs` embeds the unchanged exported dense weights plus normalization
constants. Supply `ZENSALLY_QUERY_HEAD` from `composition-query-head`. The
build-time zenpredict-bake 0.1.0 is unpublished: use the explicit Cargo patch
pointing at `composition-snapshot-baker`'s frozen owned source. The runtime
dependency is published zenpredict 0.2.0; the baker/JSON parser are build-only.
This standalone Cargo workspace does not change the shipping dependency graph.

Commands take the fixture from `composition-query-fixture`:

```
zensally-native-query-study score INPUT OUTPUT ordered
zensally-native-query-study score INPUT OUTPUT algebraic
zensally-native-query-study bench INPUT REPEATS ordered
```

Score outputs are newly created, never overwritten. The benchmark's first
warm pass reports a digest of the frozen candidate-mask selections. Timed
passes include summed-area construction and all pooled/dense query forwards;
loading, writing and candidate planning are outside timing. This is not full
image-to-crop latency. Validate every output against the ONNX reference and
the fixed three-map-pixel/action gate before interpreting performance.
