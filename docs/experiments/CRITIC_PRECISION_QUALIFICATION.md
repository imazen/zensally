# Local composition critic precision qualification

Registered after twenty continuous-pair controls fail quality suitability. Reuse
the existing pinned PhotoFramer Qwen2.5-VL-7B composition checkpoint at revision
75d7da9ffcbe794611eb69ff552bcf98de220cf4, asset manifest aab4ad8d, and the original
16-image critic plan51b78892. No new download, model, prompt, source or labels.

The previous NF4 and NF4-with-BF16-vision runs each agree with cached weak
accept/reject ordering on3/6 photographic pairs, tie2 and reverse1. Check
unquantized BF16 before expanding its suitability screen. Preserve all original
bytes and existing quantized outputs. Continue the same stateless prompt,
owned RGB sizes,512-token bound, SDPA and seed31. A derived loader config still
changes only the original null use_cache field to true.

Load BF16 with Accelerate automatic placement:12GiB GPU and8GiB CPU weight
budgets, no disk offload; runtime GPU allocation at most80 percent. Run through
run-heavy16GiB/two CPU threads, with source/model SHA256 verification before
loading. This CPU/GPU placement is an unquantized reference, not an all-GPU
bit-exact oracle or serving configuration. Record actual module placement,
dtype/quantization, complete raw responses, parse failures and resource logs.

First execute one source as a memory/loader probe, then extend the same immutable
response cache to all16 if it completes within the caps. Limit is explicit at
the caller. Partial observation timeouts never authorize restarting a live job.
No new teacher score becomes training supervision.

Compare all16 ratings and six weak-reference pair orderings with both original
quantized runs. A stronger qualification screen requires6/6 correct pair orders,
each with at least0.05 rating separation and no parse/truncation failures.
Failure is retained and prevents scale-up to the wider cached blind-review
panel under this protocol. It does not prove human artistic quality either way.
TEST and existing five quality files remain closed.16MiB new artifact budget,
cached16.6GB model weights are referenced rather than copied; API spending0.
