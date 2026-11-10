# Offline evaluation and model inventory evidence

Observed on 2026-11-08. Evaluator/inspector source commit:
`566eea2fb7f88f5705a9ba1b86e03c864b8a6814`.

Cached evaluation:

- Primary, i265: `/mnt/v/output/zensally/cached-evaluation/2026-11-08/`.
- Independent NAS via dev: `/mnt/tower/output/zensally/cached-evaluation/2026-11-08/`.
- Seven files; SHA256SUMS.json digest
  `7cf381e15554210ae3761e068fe8933dedec3268d37c866387938711d1fe0a1f`.
- Contains exact admitted manifest/selection copies, report, source, command and
  network-isolated run log. Frozen source study is unchanged.

Official geometry model inventory:

- Primary, i265: `/mnt/v/output/zensally/local-evaluation-models/2026-11-08/`.
- Independent NAS via dev: `/mnt/tower/output/zensally/local-evaluation-models/2026-11-08/`.
- 41 files; SHA256SUMS.json digest
  `4a58cd6aa5aa966bbaba3cd02506d7cdba545bde9371974421242c0effd4689b`.
- Contains 11 model artifacts, complete official HTML/link captures, actual
  resolved download URLs/headers, SHA256/byte/graph inventory, inspector source,
  package versions and execution logs. Original pre-correction inventory is
  preserved; the current scan recognizes TFL3 before appended metadata ZIPs.

Neither packet calls a paid inference API. Model binaries stay outside Git.
All members of both NAS packets passed independent streamed SHA256 readback
with `tools/crop-first/artifacts.py verify`. No R2 upload was made.
