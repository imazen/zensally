# Uncompressed automatic person-region control

Register before fitting: PCA8 full-person controls leave55/139 nominated-head
cuts; coordinate-error intervals cross zero. Test whether compression removed
useful spatial cues, retaining all47 source descriptor values. This is a
representation comparison, not newly admitted supervision or a new safety policy.

Reuse frozen person descriptor cd400d66 and all inputs, FIT-family folds,
99 FIT requests,36 paired DEV requests, nine lambdas and source weighting from
the original ridge protocol. Replace only the eight PCA components by47
identity components after FIT-only weighted normalization. The same aspect and
aspect-squared interactions yield146 design columns instead of29. Both all-
person and largest-person variants run; held families never enter transforms
or ridge fitting. Larger lambda wins ties; DEV never selects a hyperparameter.

Compare both to frozen geometry318f88d1 and corresponding PCA8 controls.
Report paired error and family intervals,139 accepted essential DEV endpoints,
all emitted/contain requests and secondary outcomes. Preserve exact native
maximum-area Minimal, strict nested Maximal, original three-grid numerical gate
and all five evaluator files. Numeric checks compare FP64 with CPU/GPU FP32.
No detector/VLM inference, API calls, new labels, TEST or runtime promotion.
Run-heavy4GiB/two threads/20-percent GPU, after verifying no large critic live.
Artifact budget16MiB; no new pixel gallery unless measured outcomes warrant it.
