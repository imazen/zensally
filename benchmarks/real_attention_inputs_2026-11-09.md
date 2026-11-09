# Real-photo attention inputs before further fitting

Local teacher sidecars and frozen candidate values complete on72 real families,
48 FIT/24 DEV. Pose is present on28/48 FIT and13/24 DEV sources; subject-gaze
input heads occur on44/48 FIT and22/24 DEV. The earlier feature pilot has4/64
and6/64 fitting-source coverage respectively.
This is a source-coverage observation, not a composition-quality gain. No paid
API calls, synthetic gradients or TEST inference supply this packet.

The226-member pinned teacher subtree is verified before use. Gaze-LLE processes
owned448-square inputs; DeepGaze III receives owned aspect-preserving inputs,
the original center prior and four six-fixation paths after an imposed central
start. Only coordinates/confidence are saved; model-internal arrays remain
transient. Tiled-interior YuNet proposals provide gaze boxes with the existing
15% width/25% height padding. Sixteen largest boxes per source are retained:
247 inputs,303 omitted. These are proposals, not human gaze or complete face
coverage. Existing pose/segment, gaze-room and scanpath scalar definitions stay
unchanged.

The exact Sol hybrid bank and frozen C002 encoder/first scorer layer produce
72×22×128×96 hidden values. The original final layer replays every cached score
exactly. Thirty-two candidate features and their source-permuted controls are
computed on recipient crop coordinates, so permutation does not substitute
different crop geometry.61 donor assignments move within the fixed role/content
groups. The134,492,896-byte cache SHA256 is
`a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3`.

No fifteen-run scalar-head experiment has started. The owner redirected work to
[label quality](crop_label_quality_2026-11-09.md) after this cache completed.
The small scalar wrapper is initialized and gradient-mask tested, not a trained
or exported model. Old quality and teacher-free student/runtime qualification
remain unmeasured for that proposed fit.

Gaze/scan jobs run concurrently under3GiB/two-thread limits. Run-heavy reports
6s/1.50GiB and52s/1.74GiB peak RSS, minimum available RAM20,855MiB. Feature caching
takes58s/1.59GiB under4GiB/two threads, minimum available RAM22,052MiB. These are
complete shared workload observations, not isolated throughput or saturation.
The first cache attempt assumes tensor donors and fails before feature writing;
its empty directory and failure log remain retained. Integer conversion keeps
the original list assignments. The earlier attention producer leaves a stale PID
after completion; separate process verification records both workers absent,
and the later producer writes terminal PID null. Original as-run source is kept.

All321 packet members/218,298,006 bytes have independent NAS membership, size and
hash readback. Local root: `/mnt/v/output/zensally/real-attention-control/2026-11-09/`.
NAS root: `/mnt/tower/output/zensally/real-attention-control/2026-11-09/`.
SHA256SUMS.json SHA256:
`86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3`.
Recheck with `just composition-verify-remote-packet dev <NAS root>`.
