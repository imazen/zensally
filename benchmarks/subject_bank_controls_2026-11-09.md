# Subject-bank learning controls — 2026-11-09

Changing proposal geometry gives a stronger weak-reference signal, but raw crops
remain visibly unacceptable. No model or shipping crop policy is promoted.
Thirty new fits complete: two banks, four arms/three seeds each, plus three
matched old-replay controls per bank. All thirty selected native replays are exact.
No API calls, synthetic-image gradients or TEST inference are added.

The [registration](../docs/experiments/SUBJECT_BANK_CONTROL.md) preserves the
same real72 sources, 48 FIT/24 DEV families, parent weak windows, C002 initializer,
94/25 old replay, 44 epochs, loss coefficient 0.25 and old-quality prerequisite.
The first control allows maximum-area choices for Maximal on the original bank.
The hybrid additionally spends up to 128 Minimal proposals on maximum-area
origins. All original caches and production semantics remain unchanged.

Across the same 394 accepted weak-reference requests, coverage is 167 original,
244 broad-Maximal and 302 hybrid. Contrastive queries become 87, 160 and 231.
Neither coverage nor a missing reference establishes artistic acceptability.

The following essential-ratio metrics average three selected seeds, balanced
by the same 17 DEV families/140 accepted-window requests. Each initializer is
evaluated using its own bank, independently of the best trainable checkpoint.

| Bank/arm | Reference match | Nominated-head cut | Old regret |
|---|---:|---:|---:|
| Original initializer |18.1232%|26.8884%|0.247001|
| Original frozen head |19.3464%|22.9739%|0.245908|
| Broad-Maximal initializer/replay-only |20.2474%|26.8884%|0.247001|
| Broad-Maximal frozen head |28.7846%|16.7717%|0.240383|
| Hybrid initializer/replay-only |17.7918%|24.9113%|0.247001|
| Hybrid frozen head |28.5294%|18.7838%|0.243295|

Broad-Maximal frozen-head learning changes match by +8.5372 percentage points,
paired family-bootstrap 95% CI [+2.7662,+15.7641], and nominated-head cuts by
-10.1167 points, CI [-19.8372,-0.7000]. Hybrid learning changes match by
+10.7376 points, CI [+2.1102,+24.0717], and cuts by -6.1275 points,
CI [-17.3120,+5.4482]. Both compare against same-bank initializer/replay-only.
Five thousand family resamples, seed31, average seeds before resampling.

All six frozen-head fits satisfy the existing old-regret prerequisite. Geometry
and permuted-image controls satisfy it in zero of twelve fits. Their match means
are 11.4029%/19.8491% for broad-Maximal and 12.1460%/15.8077% for hybrid. Hybrid
trainable arms all select epoch zero; broad-Maximal selects epochs6/6/0. The
analysis now reads explicit epoch-zero predictions so an improved trainable
checkpoint cannot silently become the initializer baseline. Reanalysis of the
original Terra study is exactly equal to its frozen result.

These intervals are conditional on repeatedly reused development families and
machine-proposed windows. They do not establish independent artistic improvement,
actual-face protection or a complete set of acceptable alternatives. The hybrid's
larger geometry coverage does not produce a better selected proxy mean than
broad-Maximal. More proposals are not automatically better learned decisions.

## Native visual and resource checks

Each bank's independently selected eight-source viewer compares actual clipped
pixels from the teacher, same-bank initializer, selected frozen head and geometry
control. Both pass 704 crop/dimension/no-stretch browser checks without JavaScript
errors. Twenty screenshots are retained; four are visually inspected: broad
3:1 Maximal and hybrid 3:1 Maximal, 1:3 Minimal, square Maximal. This is a bounded
witness review, not inspection of every saved screenshot.

The singer 424084990ff2e8d7 at3:1 Maximal still receives a crop with almost all
the face outside it, worse than the initializer. Classroom9fa07d98319d63b9 has
partial foreground/background faces in square/tall views. Dense origins improve
some placements without enforcing whole-face exclusion. These models bypass
shipping protection and do not predict contain; explicit constraints remain
necessary. Review candidates: broad seed31/epoch28 SHA256
96d27743e7edf5955f3b6b5608bdbafdd671fc9af4b550a699e0d9806ff2d08c;
hybrid seed911/epoch24 SHA256
18498fd24b5889779c735a9f7946705443716d916f0092bfba2e24a35a18e2a1.

Each twelve-fit job runs concurrently under a4GiB/two-thread cap and takes157s,
run-heavy peak RSS2.06GiB broad /2.07GiB hybrid. Replay jobs take30s/29s with
peak RSS1.89GiB/1.87GiB. Cache builders take1s each, peak RSS0.90GiB; gallery
builders2s each,1.68GiB/1.66GiB. Browser jobs5s/14s,0.23GiB/0.24GiB. These
are bounded observed durations and memory, not isolated speed or GPU-saturation
claims. Deployment runtime and allocation cost of the new bank are unmeasured.

[Earlier teacher/geometry controls](subject_choice_followup_2026-11-09.md).
[Verified evidence](subject_bank_controls_2026-11-09.pointer.md).
