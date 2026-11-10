# Cached crop evaluation, 2026-11-08

Source: `566eea2f` (network-free evaluator and model inventory). No fitting or
checkpoint selection was repeated. Inputs are the frozen current-JXL admitted
manifest and the previously selected policy's `eval-suggestions.json`.

The evaluator ran on i265 inside `sudo unshare --net`, under run-heavy with
2 GiB and one thread. It graded 352 saved crop/contain decisions from 16 canonical
VALIDATE source families, using no credentials, inference or model API. TEST
remains unopened. Input hashes, exact command and logs are in the
[artifact packet](cached_evaluation_2026-11-08.pointer.md).

| Stratum | Requests | Offered crops | Accepted / judged | Bank-positive tasks | Avoidable contains |
|---|---:|---:|---:|---:|---:|
| All | 352 | 130 | 99 / 127 | 182 | 72 |
| 3:1 | 32 | 5 | 3 / 5 | 13 | 9 |
| 1:3 | 32 | 9 | 6 / 9 | 12 | 5 |

Three offered crops have excluded/unjudged labels. Overall accepted yield is
99/352; acceptance precision is 99/127. There are zero violations of the supplied
face boxes, which does not certify detection coverage or real face safety.
Labels are noisy rendered-pixel VLM judgments, not human ratings. Bank-positive
coverage is relative to the finite cached bank, not all possible crops.

The report includes equal-family-weight estimates and 2,000 percentile bootstrap
resamples, seed 20261108, by family within each stratum. At 3:1 the precision
interval spans [0,1]; 17/2,000 resamples have no judged suggestion and are explicitly
undefined. At 1:3 it spans [0.3,1]. The small evaluation set supports identifying
failures, not a narrow quality claim.

Resource observation, not a latency benchmark:
`run-heavy: done rc=0 0s | peak-RSS 0.04GiB | min-avail 22350MiB | peak-load 13.60`.
No network API was called; there is no incremental paid evaluation usage.

Validation: evaluator fixtures cover excluded labels, exact geometry, task
completeness, family bootstrap and a positive teacher label with a protected-box
violation. Existing image-runner tests pass. No Rust inference behavior changes.
