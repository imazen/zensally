# Native reduced ORT / YuNet — 2026-11-11

Scope: native CPU inference and single-executable deployment, with embedded
weights. This does not qualify the complete face/crop pipeline, GPU providers,
code signing, notarization or older operating systems.

## macOS ARM64

Measured on Apple M6, macOS 27.2 build 26B123, 16 GiB RAM. ORT 1.31.1 source
`6af1a932e2286d5c58166085d24c68783033edee`; tools initially at `df117a88`,
subsequently preserved on main in `072a6e77`. Final executable SHA256:
`a40eab158b5029607c55db47b506ec6bdeeca01bdd8990fae5144363b9b2a2be`.

| Measurement | Stock native ORT | Reduced static executable |
|---|---:|---:|
| Runtime library bytes | 44,797,256 | Included in executable |
| Executable bytes, including embedded model | 359,008 | 3,689,680 |
| Mean forward latency, ms | 6.217689225 | 6.366729500 |

The 295,560-byte model is embedded in both probes. Reduced uses MinSizeRel;
stock uses the published package. This is a package comparison, not a controlled
attribution of size or speed differences solely to operator reduction.

Zenbench 0.1.10 completed 4/4 paired blocks, three ten-forward batches per
invocation. Paired geometric latency change is **+2.3769%** (reduced slower),
between-block log-ratio SD 0.0246552. A/A change is +0.7878%, SD 0.0145263.
Placement is explicitly unrestricted and unverified; these are exploratory
measurements without a confidence interval or performance acceptance verdict.
SSH launch/transfer are outside the native timing boundary. See
[the measurement contract](../docs/NATIVE_ORT.md).

All 134,400 FP32 output values match stock bit for bit in every paired check.
The fixed numeric gate was not relaxed. The source ONNX SHA256 is
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`;
Mac ORT-model SHA256 is
`3a733628410d886eb6e8ab82d841856f103f4a4d2972c91e0a57d76b412d2051`.

The executable ran successfully from a fresh directory containing only itself,
with system-only PATH and no DYLD overrides. `otool -L` lists only Foundation,
CoreFoundation, libiconv, libc++ and libSystem. No ORT dylib or model file is
needed beside it. This means one application file plus the operating system's
normal libraries; it is not a fully static macOS executable.

Final build resource line: `run-heavy: done rc=0 189s | peak-RSS 1.32GiB |
min-avail 8054MiB | peak-load 4.55`. Isolated inference proof:
`run-heavy: done rc=0 1s | peak-RSS 0.05GiB | min-avail 9879MiB | peak-load 2.08`;
BSD time records 52,183,040 bytes maximum resident set size. Mac resource
protection is a monitored process-tree RSS guard, not a cgroup hard limit.
The coordinator's 0.02 GiB peak for the paired SSH driver is not Mac inference
memory and must not be presented as such.

## Windows

Native x64 and ARM64 checks are being run using the manual
[`native-ort.yml`](../.github/workflows/native-ort.yml) workflow. The initial
[attempt](https://github.com/imazen/zensally/actions/runs/42933980099) failed
before compilation while Strawberry Perl's selected patch executable applied
ORT's ONNX patch. The follow-up selects Git's GNU patch explicitly. No numerical
gate or test expectation changes.

The separately generated Windows ORT model has the same 295,560-byte length and
source ONNX hash, but a different serialized-file SHA256:
`b2820464be58a69ec59704e6c655785a642645edbac6c0cb1768a26700f6ac40`.
Keep these artifact identities separate; do not infer cross-platform timing or
bit equality from their common source model.

## Retained evidence

Packet root on i265: `/mnt/v/output/zensally/native-ort/2026-11-11/`.
It retains full build logs, successful binaries, model/config identities,
standalone loader proof, paired outputs, checker results and zenbench packet.
The original failed build logs are retained too. No generated binary is committed
to Git. No paid model API calls were made for this experiment.
