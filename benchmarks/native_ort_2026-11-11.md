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

## Windows x64

The [native run](https://github.com/imazen/zensally/actions/runs/42934421925)
passed deployment, numerical and timing checks on Windows Server 2025
(10.0.26100). Zenbench identifies the hosted CPU as AMD EPYC 9V45 with four
logical / two physical assigned cores. Placement is unrestricted and unverified.
Zenbench version is 0.1.11.

| Measurement | Stock native ORT | Reduced static executable |
|---|---:|---:|
| Runtime library bytes | 17,253,728 | Included in executable |
| Executable bytes, including embedded model | 589,312 | 3,427,328 |
| Mean forward latency, ms | 13.280013333 | 14.781617500 |

Four of four paired blocks pass. Geometric paired latency change is +11.3095%
(reduced slower), log-ratio SD 0.0127380. A/A change is -1.7732%, SD 0.0140731.
All 134,400 output values match stock bit for bit in every comparison. This is
an exploratory package/configuration comparison, not an isolated operator-
reduction speed measurement. Windows memory usage was not measured.

The executable runs alone with System32-only PATH. Its imports are KERNEL32,
ADVAPI32, api-ms-win-core-path-l1-1-0 and dbghelp. It needs no ORT DLL, model file
or dynamically linked MSVC runtime beside it. Executable SHA256:
`f752e10d180dd7e08979963acac914844fc223d2dfd76aeb2e825d244ef2281f`.
The embedded ORT-model SHA256 is
`c03e1344d6ddbae7f6e0ac807ab27155165d560ae2ed6b561e6b00857b2d7e24`.
The downloaded stock DLL SHA256 matches the benchmark packet exactly:
`cb2fa518b18356617f6f9f460c012501da2819ec7e5c1d80bd8f5de20492db0c`.

The build step completed in 23 min 9 s with two compile jobs on the hosted
runner. This is an observed step duration, not a cross-machine build-speed
comparison. The original [attempt](https://github.com/imazen/zensally/actions/runs/42933980099)
failed before compilation while Strawberry Perl's selected patch executable
applied ORT's ONNX patch. Explicitly selecting Git's GNU patch corrected that
build-tool failure. No numerical gate or test expectation changed.

## Windows ARM64

The [native ARM64 run](https://github.com/imazen/zensally/actions/runs/42936770890)
passed all checks on Windows 11 (10.0.26200), with four assigned cores. The
benchmark did not report a CPU model name. Both retained executables have ARM64
PE machine type `0xaa64`; the x64 executables independently verify as `0x8664`.
Zenbench version is 0.1.11; placement is unrestricted and unverified.

| Measurement | Stock native ORT | Reduced static executable |
|---|---:|---:|
| Runtime library bytes | 17,507,640 | Included in executable |
| Executable bytes, including embedded model | 584,192 | 3,731,456 |
| Mean forward latency, ms | 24.467205000 | 24.760643333 |

Four of four paired blocks pass. Geometric paired latency change is +1.19888%
(reduced slower), log-ratio SD 0.00520559. A/A change is +0.004878%, SD 0.00194686.
All 134,400 output values match stock bit for bit in every comparison. Windows
memory usage was not measured. This is a warm, single-thread, exploratory CPU
comparison; it does not establish cross-machine performance rankings.

The executable runs alone with System32-only PATH. Imports are the same four
Windows system DLLs listed for x64. Executable SHA256:
`877ec3f5b0230a118e6fc29cf6a7f23f2691908676a70dcc75694039f6cb4308`.
The retained stock DLL matches the packet identity:
`3ec71b45c71ed057096f4914fddb379a38b9ec813794052b96fec7de81e2ea20`.

The build step completed in 34 min 25 s with two compile jobs. The preceding
ARM64 attempt progressed past dependency patching, then its assembler rejected
`/arch:armv8.2` in optional KleidiAI code. The successful build uses the upstream
`--no_kleidiai` flag and the unchanged numerical gate. The numerical checks above
qualify this CPU fallback on the diagnostic fixture, not optional KleidiAI.

The ARM64 embedded ORT-model SHA256 is
`932f72490ccddc5d7c0793af69afe2afb2172423ba1142ac97ff5a0545dfba84`.
The Mac, x64 and ARM64 serialized models share the pinned source ONNX and
operator requirements, but have distinct file hashes. Each runtime comparison
uses the same exact model bytes within its run. Do not infer cross-platform
output-bit equality from common source weights.

## Retained evidence

Packet root on i265: `/mnt/v/output/zensally/native-ort/2026-11-11/`.
It retains full build logs, successful binaries, model/config identities,
standalone loader proof, paired outputs, checker results and zenbench packet.
The original failed build logs are retained too. Windows x64 used tools at
`b870dbe6`; Windows ARM64 used `c68d3166`. The Mac packet fingerprints the SSH
worker; remote binaries, source copies and library identities are archived
separately. The later Mac stock-archive guard checks the already-verified package
and does not change compiled inputs. No generated binary is committed
to Git. No paid model API calls were made for this experiment.

The complete packet is mirrored under the same relative path on the NAS,
resolved through `dev:/mnt/tower/output/zensally/native-ort/2026-11-11/`.
`SHA256SUMS.final.json` binds the final complete evidence set. Earlier subset
manifests remain as historical snapshots. The Windows results use native CI
runners; the local workstation hostname did not resolve from the agent hosts,
and native executable launch through its SSH/WSL session remained unavailable.

Final mirror verification covers **275 member files / 286,717,204 member bytes**.
Final manifest SHA256:
`f787bf8daaeaff6a10980cb7b0f8939775fe374bf228b9b9a5f74c87a923ccad`.
All member hashes and lengths were independently checked on the NAS.
