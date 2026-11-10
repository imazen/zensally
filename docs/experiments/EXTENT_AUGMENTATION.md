# Content-preserving augmentation suitability check

Register before fitting. The matched Florence extent study shows overfitting:
small encoder final fit IoU 0.910905 versus development 0.261546; selected image
IoU 0.301438 remains below geometry0.340012 and shuffled0.369064. Test one
generalization intervention rather than increasing epochs or relaxing gates.

Keep the same source/target/initializer hashes, 48-fit/eight-development roles,
L1/GIoU loss, batch 16, 200 epochs, FP32/TF32-off, head/encoder learning rates
0.001/0.0001, weight decay 0.01, frozen BatchNorm running statistics, clip 5 and
epoch-zero selection. Run six end-to-end fits: image/permuted arms and
seeds31/137/911. Compare to the already completed unaugmented six-run matrix,
with paired family uncertainty. This is reused development evidence.

For fit minibatches only, use independent probability 0.5 horizontal reflection
within each original letterbox's content, not the complete padded canvas.
Reflect targets x0/x1 to1−x1/1−x0, preserving source-normalized coordinates and
all original content. Independently sample per-image brightness and contrast
uniformly in[0.8,1.2], contrast centered at0.5, then clamp to[0,1] and restore
black padding. No vertical flip, geometric content removal, new image, teacher
inference or changed face/crop policy. Permuted inputs receive the same
augmentation distribution with their actual input content rectangles; their
unrelated target is reflected with the same sampled flip.

Capture/restored Torch CPU/CUDA RNG state already covers these draws. Check
asymmetric letterbox padding, exact target transformation, strided input,
input preservation and color clamping before fitting. Retain all states and
histories; evaluate original, unaugmented development images. Reuse the
three-map-pixel initial numerical gate. Host cap4 GiB/two threads, CUDA35%,
state artifacts below4 GiB. No promotion, human-quality or face-completeness claim.
