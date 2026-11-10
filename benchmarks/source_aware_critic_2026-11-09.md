# Source-aware local crop-pair critic suitability

Direct comparison with whole-source context does not qualify this local
PhotoFramer checkpoint as a labeler. All12 calls parse without truncation,
but only5/12 choices match the cached weak preference and only1/6 pairs is
consistent after swapping A/B. No new labels or wider screen follows.

The [registered diagnostic](../docs/experiments/SOURCE_AWARE_CRITIC_PAIRS.md)
is separate from the failed unconditional scalar protocol. Same pinned local
BF16 model, six historical mixed-content pairs and exact owned crop RGB.
Each request adds source preview, ratio and mode, requesting only a choice
and short reason code. Historical preferences remain training-ineligible;
no human/artistic correctness claim follows from matching them.

| Cached task | Accepted as A | Accepted as B | Swap consistent | Both match preference |
|---|---|---|---|---|
|11068 square Maximal|B|B|No|No|
|11220 4:3 Maximal|B|B|No|No|
|11460 square Minimal|B|B|No|No|
|11638 square Minimal|B|B|No|No|
|12008 3:4 Maximal|B|A|Yes|No|
|12202 square Maximal|B|B|No|No|

Five tasks choose the second candidate regardless of which crop it contains,
with reason code context. The portrait consistently selects the weak-rejected
crop with reason code face. This is an observed suitability failure for this
checkpoint/prompt/task, not a claim about every local VLM or human preference.
All12 answers are18 tokens, below256. The unchanged screen requires12/12
preference agreement plus six consistent swaps and no parse/truncation errors.

An independent processor audit rebuilds all12 requests and processes all36
images separately. Each packed image block matches its independently processed
single image exactly. All six source blocks stay stable; both distinct crop
blocks actually swap. Recorded token counts and image grids replay exactly.
This rules out duplicated/unswapped processor blocks as the cause of the
observed position dependence. It does not audit every internal attention step.

Owned source previews are at most336 pixels per axis on a28-pixel grid.
Five preview aspect ratios remain exact relative to their bounded source panes;
the portrait changes0.7994791666666666 to0.8333333333333334, a relative
0.04234527687296419 difference. Both orders use the same preview; crop RGB
remains exact. This limits context fidelity, but does not explain why the
same pair chooses B in both orders on the other five sources.

## Resources and retained evidence

Run-heavy on i265, BF16 GPU10GiB/CPU10GiB resident-weight placement,
80-percent runtime GPU cap,16GiB host/two CPU threads, no disk offload:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|One-call probe|23s|12.72GiB|23653MiB|1.92|
|Complete12, reusing one response|141s|12.75GiB|23667MiB|2.44|
|Independent input audit,2GiB/one thread|3s|0.86GiB|24568MiB|2.33|

These are research-run resources, not isolated serving latency. Original
16.6GB weights and previously derived processor metadata are hash-bound
external references. The4MiB packet retains six source RGB previews,12 raw
responses and input identities, placement, protocol, full logs and independent
input audit. No download, API spending, TEST, student gradients or promotion.

Local on i265: `/mnt/v/output/zensally/source-aware-critic/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/source-aware-critic/2026-11-09/`.

Frozen27-member packet,1711932 bytes before manifest, complete NAS membership
and SHA256 readback verified. Manifest SHA256:
`fb36dfaf89d186b7f4b69e2ff6713572feb9ef8b0beae825d81062dfce7c3039`.
