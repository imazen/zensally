# Saliency experiment data provenance

No training inputs, labels, model weights or dataset roles were changed.
All corpus work in this task is validation or a container-decoding experiment.

## Frozen crop validation

Canonical imazen-26 source:
[187fbf338ce08e8e6654db7f04ddae58d5263da2](https://github.com/imazen/imazen-26/tree/187fbf338ce08e8e6654db7f04ddae58d5263da2).
Selection joins `manifests/validate.tsv` with
`manifests/imazen26_representatives_K1000_2026-06-14.tsv`, then supplements
whole validation source images to at least 50 per broad class. There are
50 photo, 50 screen, 50 line/document and 107 mixed originals. No fitting.

The checked cache on i270 is
`/mnt/v/input/zensally-crop-validation-2026-11-06/`.
`inputs.tsv` retains every PNG URL and SHA256; `source.json` records the
canonical inventory commit and both inventory hashes. Derived source sizes
never upscale originals; the fixed neural input still uses its required
256x256 preprocessing. There are 992 source-size cases and two candidates.

Input manifest SHA256:
`29788507f20a54d1324e4b22efde77ad0f6f338e8ae00350fec328f6cd408c99`.

## Native depth oracle and complete-image experiment

The portrait HEIC comes from the pinned
[Openize source archive](https://github.com/openize-com/openize-heic-net/archive/1640f6ed26dd4038b86ed9d548a7b34c2cdd7ba1.tar.gz),
`Openize.Heic.Tests/TestsData/samples/iphone_portrait_photo.heic`.
The codec-corpus cache downloads the archive; missing inputs are hard errors.
Image SHA256:
`c782e03d1ef2a709f6a84ff1b1612ef2dfa909e4872958d3975e15c79ab523c3`.
Independent libheif/libde265 depth-code SHA256 (all u16 codes, packed LE):
`4fc1e83b7bd9328f894b808c718ea94f24f2713824f4f215fd8a2de589622abc`.
The fusion experiment uses this same image with four unfitted blend amounts;
no crop-quality improvement is inferred from rectangle movement.

## Results and references

[The artifact pointer](benchmarks/w24_2026-11-06.pointer.md) records checked
local paths, verified mirrors, frozen executables, raw benchmark evidence and
report hashes. [Final measurements](benchmarks/w24_final_2026-11-06.md) identify
source revisions, commands, protocol and limits. No R2 uploads or image/binary
commits were made. Generated reports over 30 KB remain outside git.

The [arithmetic follow-up](benchmarks/w24_reductions_2026-11-06.md) reuses the
same frozen validation originals and source-size cases. No fitting, threshold
changes or corpus-role changes were introduced for reassociation experiments.

The [memory/CPU follow-up](benchmarks/w24_memory_cpu_2026-11-07.md)
reuses the same frozen validation manifest and reports 1,984 comparisons.
Both optimized runtime modes retain zero ROI/crop edge movement. No training,
labels, model weights, roles or tolerance changed. Complete raw measurement
and validation evidence is retained through the
[verified memory-study mirror](benchmarks/w24_memory_cpu_2026-11-07.pointer.md).
