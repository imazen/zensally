# TRAIN attribution and rotation join — 2026-11-09

All70,000 MIAP TRAIN IDs join the pinned Open Images information CSV, with no
missing IDs. The scan reads1,743,042 metadata rows; it acquires no image pixels
and assigns no training/development/TEST roles. All sources remain ineligible.

The638,407,721-byte source has SHA256
`05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0`;
its downloaded bytes also match the independent provider MD5. The original URL,
GCS generation, response headers, source/author/license hyperlinks, original MD5
and raw rotation are preserved in every joined record. There are18,098 distinct
canonical author-profile URLs. These are metadata group keys, not yet a split.

All70,000 records list CC BY2.0. That is the source's listed license, not a live
per-image attribution/license verification or commercial-use qualification.
Rotation fields contain59,989 zero turns,268 at270 degrees,174 at90 degrees,
28 at180 degrees and9,541 missing values. Missing rotations remain unresolved;
the five annotation-geometry quarantine IDs remain quarantined.

The initial parser recognizes integer strings but the source uses decimal
strings such as `90.0`; it consequently flags all70,000 as unknown. The
correction accepts exact finite quarter turns, preserves the raw strings and
retains the entire initial join/summary/progress separately. Invalid, fractional,
negative and nonfinite controls remain unknown. This changes no pixel geometry,
annotation bounds, source eligibility or existing test expectation.

The download/initial join takes15s with run-heavy peakRSS0.04GiB, minimum
available21,414MiB and peak-load2.66. The corrected cached join takes5s,
peakRSS0.04GiB, minimum21,499MiB and load3.08, under1GiB/one thread. These are
metadata resource observations. No training, inference, paid API call or map.

Before pixel selection: reconcile original/source/author families against
existing holdouts, freeze new roles, bound a pilot and verify downloaded-pixel
orientation against the annotation convention. Person boxes remain separate
from face-class annotations and artistic composition targets. See the
[protocol](../docs/experiments/MIAP_METADATA.md) and
[parent annotations](miap_metadata_2026-11-09.md).
