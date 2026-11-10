# Subject-choice followup — 2026-11-09

No model is promoted. Thirty matched subject-choice fits are complete: fifteen
using Terra references and fifteen using Sol references. Neither panel establishes
a reliable gain. Source-level art-direction corrections are independent of these
real-photo fits; no synthetic source has supplied a gradient in either study.

The additional authorization has settled $6.42712175 of $20: generation
$1.141760 and Responses testing $5.28536175. There are no pending reservations.
These are native usage at published rates, not invoice-verified charges.
The 212 unique cached Responses calls comprise 104 Terra, 107 Sol and one Astra.
Seven Sol and three Terra raw annotations fail structural validation; they remain
cached and included in spending. No request is automatically repeated.

## Source review

The owner rejected the skincare, watch and handbag concepts. The first skincare
prompt explicitly inserted a medical trait and the output paired it with
Brightening copy. The handbag casting inserted a wheelchair as a variety cue.
The watch appeared fused to the wrist. These are source/concept failures, not
negative labels for every possible crop.

All three originals remain archived with exact PNG-hash-bound exclusions from
positive training and evaluation-reference admission. Their three full-size
reviewed replacements preserve source families and FIT/FIT/DEV roles. The
[art-direction guide](../docs/experiments/SYNTHETIC_ART_DIRECTION.md) requires
believable scenes, appropriate person/product/copy associations, and review of
object attachment and physical geometry. The owner's tolerance for small AI
quirks does not excuse poor social judgment.

Remote ai-originals revision 7ad12a75edaf55deb5f64ce1ede4806907681c08 contains
91 ordinary JXL blobs, 123,140,805 bytes and zero LFS pointers. Every stored image
hash and all three rejection/replacement bindings verify. This counts archived
images, including rejected originals, not 91 approved training examples. The
active new round remains 32 sources: twenty commerce, sixteen white-background,
ten at 4:3/3:4; twelve candids. Three extra replacement files do not add families.

## Sol control

The 104-source Sol-primary panel reuses nine cached requests and makes 95 new
POSTs. It finishes with 97 structurally valid responses. Three replacement
sources receive separate valid Sol responses. The real-photo subset preserves
the same 48 FIT and 24 DEV families; 67 of 72 responses are structurally valid.
Its image, content, candidate boxes and eligibility tensors are bit-identical
to the Terra panel. Source roles and TEST remain unchanged.

Sol provides 525 accepted reference queries; only 200 have a compatible old-bank
window, and 102 also have explicit supplied-head-cut negatives. Its 1,206 valid
decisions include 697 crop, 38 unchanged and 471 contain. Unknown candidates get
zero gradient from the new acceptable-set term. Complete head coverage and
artistic correctness remain unreviewed.

The same four arms, three seeds, 44 epochs and three matched replay controls
use the frozen C002 initializer, 256-square owned letterboxing, AdamW 1e-4,
weight decay 1e-4, clipping 5, FP32/TF32 off and the existing old-quality gate.
All fifteen selected native replays are exact. Full optimizer/RNG and terminal
states remain retained. Accepted-essential-window evaluation covers 17 DEV
families and 139 requests; it is a development diagnostic, not a new TEST.

| Arm | Reference match | Supplied-head cut | Old regret | Old gate |
|---|---:|---:|---:|---:|
| Initializer/trainable selected at epoch zero |11.7066%|41.6773%|0.247001|3/3|
| Frozen image encoder |11.9867%|40.1112%|0.240593|3/3|
| Matched old replay only |11.7066%|41.6773%|0.247001|3/3|
| Geometry only |9.3583%|39.8878%|0.875730|0/3|
| Permuted images |11.2257%|36.6845%|0.409883|0/3|

Frozen-image minus initializer/replay: match +0.2801 percentage points,
95% family-bootstrap CI [-0.5602,+1.4006]; supplied-head cut -1.5661 points,
CI [-4.0591,+0.6774]. Both intervals include zero. The panels have different
accepted-window denominators, so their absolute cut rates are not a direct
Terra-versus-Sol comparison.

The twelve Sol fits take 87 seconds, run-heavy peak RSS 2.05 GiB; three replay
controls take 31 seconds, peak RSS 1.89 GiB. They run concurrently under separate
4 GiB/two-thread caps; these are observed job durations, not isolated speed
comparisons. No deployment latency or GPU saturation claim follows.

## Common-panel and geometry controls

Cross-teacher evaluation applies the initializer and both selected frozen heads
to identical model inputs and each teacher's own ROI panel, including contain
requests. On the Sol panel's 22 families/308 essential requests, raw nominated-
head cuts are 44.8052% initializer, 40.5844% Terra head and 42.2078% Sol head.
More Sol annotations have not made this trained head better on that common panel.

A separate diagnostic requires one whole nominated head and forbids partial
intersection with any nominated head. It offers a crop for 86.6883% of that Sol
panel, with zero nominated-head cuts by construction. Others return contain.
It may exclude other people wholly. These unreviewed nominated ROIs are not
complete actual-face detection; no shipping guard or coverage requirement changes.

The candidate-bank audit uses 394 accepted Terra queries and makes no API call
or fit. Coverage means an eligible, geometrically compatible rectangle within
three source-normalized 128-grid pixels per edge of a finite proposed window.
It does not classify all other compositions as bad.

| Geometry | Minimal coverage, 199 queries | Maximal coverage, 195 queries |
|---|---:|---:|
| Original shared bank and eligibility |141|26|
| Original bank, allow maximum-area Maximal choices |141|103|
| Mode-specific 128 bank, same broad Maximal choice |199|80|
| Mode-specific 256 bank, same broad Maximal choice |199|88|
| Offline same-call ROI anchors, 128 budget |199|89|

Strict Maximal eligibility on the new 128 bank reaches only 10/195. Thus much
of the apparent Maximal gain comes from eligibility semantics, not more proposal
density. A hybrid with dense maximum-area Minimal origins and the existing broad
Maximal grid is the next controlled research candidate. Neither density nor ROI
anchors have measured deployment speed or artistic-quality benefit. Existing
bank semantics and tests stay frozen for comparison.

[First Terra study and visual failures](subject_choice_fitting_2026-11-09.md).
[Followup evidence](subject_choice_followup_2026-11-09.pointer.md).
