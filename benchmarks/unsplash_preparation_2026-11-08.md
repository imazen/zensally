# Unsplash corpus acquisition and owned previews — 2026-11-08

All144 source JPEGs from the user's dataset repository are acquired and every
owned working PNG reads back independently to exactly matching RGB. Roles
remain unassigned and no new images supply gradients or artistic labels.

Source: [imazen/zensally-datasets](https://github.com/imazen/zensally-datasets),
`unsplash-originals` at`b8d7cd87562c271a7725bd446d656587a5886c01`.
All144 originals match pinned Git blob SHA1, manifest SHA256 and encoded size:
479,245,243 bytes. Source URLs, provider IDs, attribution and repository metadata
are retained. The local dataset checkout contains another session's changes
and LFS pointers; it was inspected read-only and never synchronized or changed.
The isolated cache downloads actual pinned remote Git blobs.

There are124 provisional photographer groups, zero exact source-hash duplicate
groups and zero exact original/encoded/provider-path identity matches against
the consulted canonical/historical metadata. Perceptual/scene duplicate auditing
is incomplete; this is not a complete no-overlap claim. No family split is yet
assigned. The source cache is on i265 at
`/mnt/v/input/zensally/unsplash-screen/2026-11-08/`.

Every source header is independently observed as RGB and orientation1. The
largest is11,339×14,173,160,707,647 pixels. A caller-selected200MP budget was
lost during the pipeline's JPEG header probe, which still rejected at120MP.
The owned zenpipe fix`0b3274f8ce661333af5bbdb3c0d5e5a2f00fa1f7` propagates
explicit probe budgets while preserving default/tight rejection and registry
gates. The standalone consumer pins that fix and the qualified source graph;
no codec/resizer kernel or new public Rust API is introduced.

Working previews use strict-sRGB color management, Lanczos3 downsampling and a
long edge of1280 with preserved aspect ratio and no upscaling. Original/native
dimensions remain separate from preview dimensions. Neural inputs derived from
these previews must identify the new preprocessing contract; they are not
bit-equivalent to earlier direct-native256 inputs. The actual144 PNG+RGB views
occupy686,192,117 bytes. All144 independent PNG shape/orientation/sample
readbacks pass. Working-manifest SHA256 is
`9d7f1f44c092084391b31a172a8253536686b2b0ec3fca9f5e1250164e4b6cda`.

The largest actual photo yields1024×1280,1,188,377 encoded PNG bytes. The
shared codec byte budget includes decoded output, so the adapter allows800MB
there and separately bounds final encoded previews to16MiB. Encoded input is
fallibly bounded to64MiB, pipeline dimensions to200MP/32768 and working views
to2GiB total. Process memory is capped at6GiB; no library default is raised.
Failed local-LFS and earlier decode/probe attempts remain retained.

run-heavy on i265: acquisition169s, peakRSS0.05GiB, min-available23,441MiB,
peak-load15.27. The complete144-preview/readback job takes70s,
peakRSS0.95GiB, min-available23,373MiB, peak-load14.30. The pinned release
consumer build takes13s/0.40GiB; strict Clippy takes7s/0.55GiB. These are
preparation costs, not inference throughput. API model calls/cost remain zero.
