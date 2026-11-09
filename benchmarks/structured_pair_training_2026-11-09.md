# Structured nested pair training

All nine matched fits fail the unchanged original quality gate. Training a
strict-pair objective improves some nested decisions but does not preserve
original quality. The separate capacity probe learns more FIT pairs while DEV
quality worsens. No model, label interval or runtime policy is promoted.

The [registered protocol](../docs/experiments/STRUCTURED_PAIR_TRAINING.md)
uses94 original FIT/25 DEV and48 newer FIT/24 DEV sources. Family roles remain
fixed and TEST remains closed. Start from4b801ad4 with frozen image encoder,
first scorer and scalar base; train only the96→32 ReLU→32 residual probe.
Replay, independent-endpoint top-choice and strict-pair top-choice objectives
each run seeds31/137/911 for32 epochs plus zero. Every epoch, optimizer, RNG,
newer DEV scores and old raw/paired choices is retained. There are1024 original
FIT ratio requests with both endpoint targets valid; no missing target is invented.
No new paid API calls or teacher judgments.

## Original quality

Original25 DEV has448 valid essential endpoint targets. Lower regret is better;
both raw and joint nested output must pass the original0.24700142443180084 limit.
The initial model scores0.244554802775383 raw and0.2601603865623474 nested.
No epoch in any of the nine fits passes both. Only explicitly named
`best-ineligible.pt` files are retained as diagnostic choices; no `selected.pt`
is produced. Best paired regret within each arm, descriptive rather than a
deployable selection:

| Arm / seed | Epoch | Full CUDA raw regret | Full CUDA paired regret | Both gates |
|---|---:|---:|---:|---|
| Replay /911 |9|0.2561699151992798|0.2501763105392456|fail|
| Independent top /911 |31|0.2609446346759796|0.251709520816803|fail|
| Paired top /911 |10|0.26410841941833496|0.24835282564163208|fail|

Full CUDA evaluation uses the five unchanged quality/model implementation
files from9f12692417d774653ed676d236e0f89eaafa7385 and reproduces the original
baseline exactly. All nine chosen ineligible checkpoints fail both-gate qualification.
The newer DEV raw weak-head cut rate also worsens for every chosen checkpoint;
these diagnostics supply no artistic labels or complete-face claims.

## Capacity versus generalization

After the matched failures, a separate FIT-only capacity probe removes the
competing old-rank and newer acceptable-set replay losses. Seed31, pair cross
entropy alone, AdamW0.003, eight passes over1024 paired FIT requests, batches16.
DEV does not select or stop this run. Nine epoch-boundary states are retained.

| Diagnostic | Initial | After eight passes |
|---|---:|---:|
| Exact frozen target pair agreement, FIT |196/1024|433/1024|
| Essential paired regret, FIT |0.12124394625425339|0.07687820494174957|
| Essential paired regret, DEV |0.2601603865623474|0.34160929918289185|
| Essential raw regret, DEV |0.244554802775383|0.3619333803653717|

No epoch passes both DEV gates. This demonstrates a learnable FIT signal and
poor generalization under this probe; it does not prove the architecture can
fully memorize the targets or cannot improve with another design. It does not
establish that the weak target pairs are artistically excellent. The preceding
target-only oracle remains a diagnostic, never a runtime selector.

## Numerical checks

For each of nine chosen checkpoints:

- Compact residual arithmetic:1309 original FIT/DEV pair requests, zero endpoint
  index changes, zero edge movement or three-grid gate violations.
- Newer72 FIT/DEV raw endpoints:1584 queries, zero compact choice changes.
- Full CUDA encoder/query on25 original DEV sources:275 paired requests,
  zero endpoint index changes or edge movement against cached CPU inference.
- Tensor selection agrees with the existing exhaustive Python selector on all
  275 original DEV pairs. Separate small-bank tests exercise ties, masked zero
  slots, missing pairs and finite training gradients.

Largest compact score difference across all nine models and both caches is
2.86102294921875e-6. Pair sums use binary64 values of FP32 scores and ascending
index ties. These checks qualify numerical behavior, excluding preprocessing,
detectors, newer paired selection and artistic/face safety. No latency claim.

## Resources and retained evidence

i265, run-heavy, two CPU threads,4GiB hard memory cap:

| Run | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Nine fits |30s|1.33GiB|21969MiB|
| Full CUDA quality / arithmetic |7s|2.19GiB|21218MiB|
| FIT-only capacity probe |6s|1.42GiB|22125MiB|

These are tool durations and process RSS, not serving latency or deployment
memory. GPU was resolved idle before qualification; no API spending increased.

Frozen packet:345 members,109938904 bytes before identity/manifest,256MiB cap.
Manifest SHA256 `ec53a3fb43281f0b7860a853b225149d0600fb67032f9cfb5f7a6cab9bd7853a`.
On i265: `/mnt/v/output/zensally/structured-pair-training/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/structured-pair-training/2026-11-09/`.
Full membership/hash readback verifies the mirror separately. Protocols,
training histories, every epoch state, predictions and resource logs remain there.

Next design question: whether a scorer that explicitly models the outer/inner
relationship improves held-out composition, rather than summing independently
scored endpoints. Additional fixed-head optimization has not established that.
