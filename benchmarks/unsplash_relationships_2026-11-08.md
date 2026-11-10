# Acquired-photo relationship screen — 2026-11-08

All144 owned working RGB previews are fingerprinted with the reused zensimx
exposure-normalized linear-luma kernel. All10,296 unordered pairs are scored;
none reaches the registered Hamming-distance≤10 flag. The minimum observed
distance is13. This does not prove absence of edited/cropped relatives.

The conservative photographer barrier has124 components:108 singletons,
13 pairs, two triples and one group of four. No fingerprint edges merge
different photographer groups. No source role or crop label is assigned by
this screen, and no canonical holdout pixels are opened.

The first source's tight, padded and short-final-row RGB representations
produce the identical fingerprint`dae6bddfb7b6fffd` and positive luma peak
0.87039125. The original exposure/unrelated/black/nonfinite/geometry controls
and new strided frontend control pass. The hash representation is a new
working-preview era, not interchangeable with earlier native-source hashes.

Origin: imazen/zensimx`ed956176e02e335d4a6efb3c31724e2aaf361d34`, with exact
source hashes retained in ORIGIN.json. No fingerprint/resampling kernel is
reimplemented. The corpus-preview research tool's source graph is pinned;
its additional fingerprint binary does not change frozen preview artifacts.

run-heavy on i265: tests/strict-Clippy/release build14s, peakRSS0.65GiB,
min-available23,522MiB, peak-load13.60. The complete source/pair screen takes2s,
peakRSS0.03GiB, min-available24,482MiB, peak-load13.77, under2GiB/two-thread
bounds. Only scalar hashes, distances, identities and bounded stride fixtures
are saved; no diagnostic planes, training gradients or paid model calls.
