# Expanded crop study artifacts

Frozen 2026-11-08. [Findings and limits](crop_expanded_2026-11-08.md).

- Evaluation host: i270.
- Inputs: `/mnt/v/input/zensally-crop-expanded-2026-11-07/`.
- Independent input mirror: `/mnt/tower/input/zensally-crop-expanded-2026-11-07/`.
- Input payload: 35 files; `SHA256SUMS.json` SHA256
  `51ebd405ffea02d2de90210e7b5f13e97c22242b512076bc9abe289819c511fc`.
- Outputs: `/mnt/v/output/zensally/crop-first/2026-11-07-expanded/`.
- Independent output mirror:
  `/mnt/tower/output/zensally/crop-first/2026-11-07-expanded/`.
- Output payload: 2,055 files; `SHA256SUMS.json` SHA256
  `ee5ef7b72408c67f268b857d46eeea6f6729112ca6644333784b0954bf9fa974`.
- Served readback: the same output path on dev.

Review gallery:

http://dev.lan:3300/zensally/crop-first/2026-11-07-expanded/index.html

Separate candidate-coverage audit:

http://dev.lan:3300/zensally/crop-first/2026-11-07-expanded/coverage-audit/index.html

Verify complete membership and hashes using
`python3 evidence/annotation-source/artifacts.py verify ROOT` from a retained
copy. Resolve the mirror mount with `ssh dev findmnt -T /mnt/tower` rather than
assuming its backing storage. HTTP readback must match the frozen index hash.

Readback verified 2026-11-08: all 2,055 output payload files match on both
the independent mirror and served copy. HTTP returns matching hashes for both
gallery indexes, both summaries, the cost summary and two candidate PNGs;
the HTTP membership-manifest hash also matches. All 35 input payload files
were independently verified before annotation. Both browser suites pass.

Source/proposal/judgment manifests retain immutable roles, raw bounded prompts
and responses, all preview PNGs, URL records, independent pixel/LFS oracles,
failed attempts, per-batch execution logs, token/cost ledgers, compiler identity,
source snapshots and frozen executables. Private full CLI context is excluded.
No image, binary, model weight or large generated packet is committed to git.

Frozen renderer SHA256:
`919de5322bd466678ab9f631fba269418db34c165b460735b8e92f279feb5ece`.
Standalone zenresize preview worker SHA256:
`ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041`.
Annotation/report tooling landed through `53efc439`; the packet also retains
the exact source files and both resolved Cargo lockfiles. Runtime/model payload
measurements from other studies are not assigned to this untrained crop student.
