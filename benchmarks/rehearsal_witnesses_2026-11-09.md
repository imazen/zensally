# Selected student's actual crop review

The quality-preserving contrastive student has useful framing repairs and
visible regressions. Its old ranking-quality pass and faster neural runtime
do not justify promotion. Face/head protection and artistic composition remain
unresolved in these raw ranker decisions; shipping guards are separate.

Seed31 changes90 essential DEV choices versus replay-only:11 remove a weak
nominated-head-cut label, two add one and77 change without changing that label.
The fixed diagnostic selector takes two improvements, both regressions and one
other change, largest edge movement first with distinct source families.
This deliberately selected five-family panel is not a representative quality
sample or an independently held-out evaluation set.

Each source displays the frozen original, replay-only, contrastive, real-feature
and permuted-feature outputs, plus containment. All24 distinct PNGs representing
the35 displayed panes are inspected; exact PNG hashes identify duplicate panes.
These are Codex observations, not human gold or new training labels:

- [Play-structure scene](https://www.flickr.com/photos/mpeinadopa/3941433819),
  a1d2f44a45a19cbd,9:16 Maximal: the contrastive output keeps the person inside
  the frame while replay-only clips them at the right edge. The original
  retains the nominated head but clips more body at the left. The person faces
  away; this is head/body framing evidence, not visible-face recall.
- [Waterfall](https://www.flickr.com/photos/bnphotos400/7591424414),
  48cd9b5fa18a01d8,3:4 Minimal: the shifted crop retains both heads and the
  waterfall; original/replay-only put the right head across the edge. Existing
  body truncation at the source bottom remains inherited.
- [Group at a counter](https://www.flickr.com/photos/projetoluzevida/15328959047),
  9b15ccd240edae57,1:1 Maximal: contrastive cuts through the left person's head
  and upper body at the edge. Original/replay-only retain the selected central
  head. Real/permuted feature outputs share the contrastive pixels here.
- [Baseball scene](https://www.flickr.com/photos/mathplourde/4846668094),
  ff5b19334d988f53,3:1 Minimal: replay-only retains the foreground subject's
  full body; contrastive returns the original upper band, truncating the lower
  body and leaving background body/head fragments. Main-face presence does
  not make the overall framing satisfactory.
- [Two people with signs](https://www.flickr.com/photos/www_ukberri_net/3312007362),
  f615a0f4c139618e,3:4 Minimal: contrastive chooses the right person/sign
  instead of the left. Both selected visible faces are retained, with fragments
  of the excluded person. No requested story establishes a universal winner.

An independent complete-neural replay checks25 Torch CUDA model decisions
(five arms x five scenes) and ten ORT decisions (original/contrastive):35/35
match the displayed cached-head indices exactly, zero edge movement and zero
three-grid violations. This binds these observed failures to actual neural
outputs, not stale or different model predictions. Checkpoints are retained
with exact hashes; the complete82-member rehearsal parent packet verifies.

Owned RGB/cropping/resizing/PNG encoding uses the frozen executables. Independent
readback checks every generated PNG against expected RGB, and source PNGs
against their owned RGB. Canonical DEV family identities, original dimensions,
selected native boxes, working-preview bounds, metadata/author/license links
and weak labels remain in the viewer. No diagnostic maps are generated.

Assets including retained RGB total57,184,127 bytes under the128MiB budget.
Rendering takes1s/run-heavy peak RSS0.63GiB, min available23114MiB,2GiB/two
threads. Complete-neural decision verification takes1s/1.32GiB under4GiB/two
threads. These are research tasks, not serving latency or memory estimates.
The served page and all35 PNGs match local bytes over HTTP; PNG bytes total
21,129,215 and decode independently. Browser layout is not claimed tested.

Evidence: `/mnt/v/output/zensally/rehearsal-witnesses/2026-11-09/`.
No API calls, new gradients, TEST inference, new quality admission or model
promotion. Existing purpose/face safeguards and the zero excellent nested-range
baseline stay unchanged. Further work must separate valid subject choice,
hard face/head admissibility and composition preference rather than treating
a lower aggregate head-cut rate as artistic success.
