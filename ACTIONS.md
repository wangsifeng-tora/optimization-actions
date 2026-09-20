# Successful vLLM ROCm optimization actions

Source: the issues listed in [`CHECKLIST.md`](./CHECKLIST.md), plus the merged PRs and measured follow-ups referenced by those issues.

**Meaning of “successful”:** merged changes, or a candidate/workaround with an explicit correctness check and a reported performance result. A result from gfx950, gfx1100, or another model is evidence of a technique—not a promise that it wins on MiniMax-M3/gfx942.

The **Impacts current MiniMax-M3 FP8 path?** column refers to the inspected `mi325x/e2e/best` launch: 8× MI325X/gfx942, TP8/PP1/DP1, EP off, vLLM `0.26.0+rocm723`, AITER dense/sparse attention, Triton indexer, FP8 main KV, shared-expert fusion, INT4 QuickReduce, and DSpark with draft `TRITON_ATTN`. `Yes` is reserved for an exact source/config change that is reachable on that path; conceptual similarity, another model, another GPU, or a future/absent PR is `No`.

## Attention

### Sparse attention, MLA, and indexer

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Replace the MiniMax fresh-decode indexer path with request-balanced score tiling, shared K tiles across query rows, and a fused top-k + sparse page-table Triton path; keep a graph-safe guarded fallback. | [#54681](https://github.com/vllm-project/vllm/issues/54681), [PR #54682](https://github.com/vllm-project/vllm/pull/54682) | Merged. TP4 gfx950 full-response ITL improved 9.8–14.3%; kernel tests covered index-head counts 1/2/4 and CUDA-graph replay. | The most direct precedent. Port/benchmark the same shape- and architecture-guarded idea on gfx942 rather than assuming gfx950 tuning transfers. | No |
| Remove full-pool BF16 materialization from sparse decode: read the quantized cache directly in the Triton decode path; use a caller-sized chunk for prefill gather. | [#41962](https://github.com/vllm-project/vllm/issues/41962), [PR #41812](https://github.com/vllm-project/vllm/pull/41812) | Merged before this search window; the follow-up confirms `rocm_forward_decode_fallback` and `rocm_dequantize_blocked_k_cache` were removed. | Memory-safety/capacity baseline for long-context M3; check that the deployed image contains this path before profiling. | No |
| Eliminate per-decode sparse-MLA fill/fill-functor launches in the hot loop. | [PR #44527](https://github.com/vllm-project/vllm/pull/44527), tracked by [#57406](https://github.com/vllm-project/vllm/issues/57406) | Listed as completed ROCm work. | Low-risk launch-overhead check for M3 decode; verify the current image already includes it. | No |
| Enable uniform-batch CUDA-graph mode for ROCm sparse MLA, including MTP-shaped batches. | [PR #45149](https://github.com/vllm-project/vllm/pull/45149), tracked by [#57406](https://github.com/vllm-project/vllm/issues/57406) | Listed as completed ROCm work. | Relevant when M3 uses speculative verification and uniform graph capture; validate graph correctness on gfx942. | No |
| Select decode KV split counts with a work-per-split heuristic instead of a fixed cap. | [PR #46832](https://github.com/vllm-project/vllm/pull/46832), tracked by [#57406](https://github.com/vllm-project/vllm/issues/57406) | Listed as completed ROCm work. | Direct long-context decode candidate; tune against M3’s sparse block budget and actual context distribution. | No |
| Skip redundant sparse-index remapping on non-indexer layers. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51309](https://github.com/vllm-project/vllm/pull/51309) | Reported as an approximately 1% TTFT action; in review in the tracker. | Prefill-only/TTFT candidate; likely orthogonal to M3’s decode indexer optimization. | No |
| Skip clearing sparse-prefill MQA logits when the consumer is bounded by real sequence lengths. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51314](https://github.com/vllm-project/vllm/pull/51314) | Reported as an approximately 1% TTFT action; in review in the tracker. | Test on M3 prefill; do not reintroduce a clear for graph safety without proving the consumer needs it. | No |
| Fuse the DSA indexer prologue: K normalization, Q/K RoPE, FP8 quantization, and K-cache write. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51315](https://github.com/vllm-project/vllm/pull/51315) | Reported as approximately 1% end-to-end throughput; in review in the tracker. | Useful only if M3’s indexer uses the same prologue and dtype/layout; inspect dispatch before porting. | No |
| Make gfx950 top-k split count and 1024-thread policy depend on live row length; add a calibrated `topK=2048` policy. | [#55327](https://github.com/vllm-project/vllm/issues/55327) | 25% median and up to 64% kernel-time reduction for GLM-5.3; exact results and no slower measured shape; candidate not yet merged. | gfx950-specific, so not a direct gfx942 action. The transferable action is shape-aware split-count tuning, not the constants. | No |
| Stop clearing the full `max_model_len` sparse-indexer logits buffer on every decode step; pass the equivalent of CUDA’s `clean_logits=False` contract. | [#55373](https://github.com/vllm-project/vllm/issues/55373) | Two A/B runs reported +16.8% throughput with unchanged needle/GSM8K behavior; candidate branch, not yet merged. | High-priority M3 check if the gfx942 path still fills a full-context buffer. Confirm every consumer is bounded before applying. | No |
| Use the AITER top-k kernel at long context once the installed AITER version supports the shape, instead of retaining the old `>64K` fallback. | [#55615](https://github.com/vllm-project/vllm/issues/55615) | AITER 0.1.21.post1 was 20.5–24.1% faster than the vLLM top-k kernel with identical indices; issue remains open. | Version- and architecture-gated candidate. Measure the actual AITER version on gfx942; do not simply delete the guard. | No |
| Size the GLM sparse-indexer decode workspace from the actual decode bound/capture bound rather than `max_num_batched_tokens`. | [#57406](https://github.com/vllm-project/vllm/issues/57406), [PR #57701](https://github.com/vllm-project/vllm/pull/57701) | Tracker marks it merged; PR title reports 3072 MiB saved. | Check whether the same workspace formula applies to M3. This is capacity/maximum-concurrency optimization, not per-token speed. | No |

### MLA preparation and backend selection

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Fuse decode QK-RoPE, Q concatenation, KV concatenation, and KV-cache write through AITER. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [PR #47757](https://github.com/vllm-project/vllm/pull/47757) | Tracker reports the action as in review with an expected 2–4% throughput gain. | Only applies if M3’s MLA layout and cache write match; inspect backend selection first. | No |
| Use `ROCM_AITER_UNIFIED_ATTN` with AITER enabled instead of allowing the generic `ROCM_ATTN` choice on long-context AMD serving. | [#56945](https://github.com/vllm-project/vllm/issues/56945) | On MI300X/MI325X/MI355X, AITER + unified attention improved Qwen3.5 decode from 1.7× to 3.7× over the stock path as context grew. | A configuration A/B, not a universal default. Test M3’s sparse MLA backend separately; do not force unified attention over its model-specific backend. | Yes |
| Prefer BF16 KV over FP8 KV when the GPU/backend has expensive FP8 conversion and no native FP8 path. | [#56992](https://github.com/vllm-project/vllm/issues/56992) | On gfx1100, BF16 reduced context-slope cost 2.2× and improved 2K decode throughput 21.6→29.1 tok/s; platform-specific. | gfx942 is not gfx1100. Keep as a diagnostic A/B only; M3’s current FP8 KV result must be measured independently. | No |

## MoE

### Routed-expert kernels and routing

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Use tuned AITER GEMM for the MoE router gate instead of the generic ATen GEMM path. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [PR #50535](https://github.com/vllm-project/vllm/pull/50535) | Tracker reports approximately 3% end-to-end throughput improvement; in review. | M3’s router dtype/shape differs from GLM; check whether the exact `(M,N,K)` route is supported before enabling. | No |
| Tune block-FP8 fused MoE for low-batch decode. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46642](https://github.com/vllm-project/vllm/pull/46642) | Completed GLM-5.2 optimization action. | Relevant to decode-sized M3 batches if its FP8 MoE backend uses the same kernel family. | No |
| Vectorize the FP32 `moe_sum` reduction and support arbitrary top-k. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46643](https://github.com/vllm-project/vllm/pull/46643) | Completed GLM-5.2 optimization action. | Check M3’s top-k=4 reduction path; likely a small launch/elementwise win. | No |
| Restore the fast MoE reduce-scatter path by removing an additional communication step; add sequence-parallel support where it improves the shape. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PRs #46635, #47070, #48763](https://github.com/vllm-project/vllm/pull/48763) | Tracker records 1.9–5.0% sequence-parallel gains and ~5% recovery from the reduce-scatter regression. | Relevant only to the chosen M3 TP/EP/DP layout; compare collective count and payloads rather than copying the parallelism setting. | No |
| Retune MXFP4 MoE for gfx950. | [#57230](https://github.com/vllm-project/vllm/issues/57230), [ROCm/aiter #5045](https://github.com/ROCm/aiter/pull/5045) | Tracker marks the retuning done, with approximately 7% fused-MoE operator improvement. | M3’s dynamic FP8 routed experts are not MXFP4; no direct transfer unless the checkpoint/backend changes. | No |

### Shared experts and expert gates

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Fuse the shared-expert gate’s skinny GEMM, sigmoid, broadcast, and multiply into one Triton kernel with a shape-guarded fallback. | [#43187](https://github.com/vllm-project/vllm/issues/43187), [PR #43190](https://github.com/vllm-project/vllm/pull/43190) | MI355X Qwen3-Next A/B: +4.65% balanced, +6.36% decode-heavy, +10.29% prefill-heavy mean throughput. | Apply only if M3 has the same scalar shared-expert gate and the AITER fused-shared-expert path is not already taking over. | No |
| Enable fused shared experts for block-quantized FP8, wire the main GLM5Next MoE and its MTP block, and derive the weight mapping from the built module’s fusion flag. | [#54376](https://github.com/vllm-project/vllm/issues/54376), [PR #53097](https://github.com/vllm-project/vllm/pull/53097) | +9% at one user and +19% at 16 users on 8× MI350X; correctness/needle checks unchanged; fixed and merged. | Strong pattern for M3: verify the FP8 compatibility gate, shared-expert weight mapping, and MTP module independently. | No |

## FFN, normalization, and elementwise overhead

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Register the Triton W4A16 GEMM and its fake/custom op so compile mode does not silently lose the intended kernel. | [#49699](https://github.com/vllm-project/vllm/issues/49699), [PR #51453](https://github.com/vllm-project/vllm/pull/51453) | Issue follow-up identified missing custom-op registration; after registration decode throughput reached 15.7 versus the degraded path. | Relevant as a compiler/backend sanity check for any quantized FFN path; not an M3-specific W4A16 change. | No |
| Add non-contiguous RMSNorm support to the fast kernel path. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PR #49750](https://github.com/vllm-project/vllm/pull/49750) | Tracker title reports 1.2–3.1× kernel improvement. | Check M3’s residual layout and whether it currently falls back because of contiguity. | No |
| Remove redundant clone/copy operations in GLM/DeepSeek model paths. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46651](https://github.com/vllm-project/vllm/pull/46651) | Completed GLM-5.2 action. | Cheap source audit for M3’s analogous residual/attention paths; validate graph aliasing before removing copies. | No |
| Dequantize MXFP8 weights once after loading when `dot_scaled` cannot consume the original shape, instead of rebuilding BF16 weights every forward. | [#56506](https://github.com/vllm-project/vllm/issues/56506), [PR #56560](https://github.com/vllm-project/vllm/pull/56560) | 4.6% lower median TPOT at concurrency 1 and 2.3% at 8; dequantization fingerprint disappeared. | M3 uses a different FP8 format in the current report, so treat this as a kernel-pattern check, not a direct patch. | No |

## Speculative decoding

### Draft attention and multi-step metadata

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Override DFlash2 draft attention from `ROCM_ATTN` to `TRITON_ATTN` when the ROCm prefix-attention path loses acceptance under batching. | [#53323](https://github.com/vllm-project/vllm/issues/53323) | At concurrency 4, output throughput improved 184→421 tok/s and acceptance 6.76%→57.05%; correctness path was stable. | Backend-selection experiment for M3’s draft model; do not assume the target and drafter should use the same backend. | No |
| Opt sparse-MLA/indexer metadata builders into fused multi-step draft decode with an empty update hook when positions/lengths are already advanced by the framework. | [#54369](https://github.com/vllm-project/vllm/issues/54369) | Acceptance stayed ~3.05; throughput improved 2.7% (2962→3043 tok/s) in the reported run. | Directly relevant to M3’s DSpark/MTP path; check whether the remaining per-step cost is MoE verification rather than metadata rebuild. | No |
| Skip the sparse indexer top-k work for short dense-MHA layers in MTP cases. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PR #50904](https://github.com/vllm-project/vllm/pull/50904) | Tracker title reports 2× kernel-level improvement. | Check M3’s layer schedule: only apply where the layer is provably dense/non-indexer and the logits are not consumed. | No |

## Communication and parallelism

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Fuse QuickReduce with RMSNorm on ROCm. | [#57149](https://github.com/vllm-project/vllm/issues/57149), [PR #48249](https://github.com/vllm-project/vllm/pull/48249) | Merged; tracker estimates approximately 20% TTFT improvement. | M3 already has a QuickReduce/INT4 baseline in the local benchmark; confirm the fusion is enabled and avoid double-counting it as a new action. | No |
| Replace MoE all-reduce with reduce-scatter and remove an extra communication step in the regression fix. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PRs #46635, #48763](https://github.com/vllm-project/vllm/pull/48763) | Completed GLM-5.2 action; ~5% end-to-end recovery reported for the regression fix. | Compare against M3’s actual TP/EP collective trace; the best operation depends on group size and topology. | No |
| Use virtual-batch PCP/DCP-style MLA parallelism to reduce replicated KV-cache pressure. | [#46654](https://github.com/vllm-project/vllm/issues/46654), [PRs #46570, #46076](https://github.com/vllm-project/vllm/pull/46076) | Completed NVIDIA/FlashInfer-side GLM work. | Not a drop-in ROCm action. The later DCP RFC explicitly reports that AITER FP4/FP8 MLA query replication is not implemented on gfx942. | No |

## Runtime, graph, and configuration hygiene

| Action / modification | Source | Evidence reported in the issue | MiniMax-M3 / gfx942 relevance | Impacts current MiniMax-M3 FP8 path? |
|---|---|---|---|---|
| Add an `AITERConfig` object to `VllmConfig`, preserving existing environment-variable defaults while making AITER toggles explicit and testable. | [#53938](https://github.com/vllm-project/vllm/issues/53938), [PR #54474](https://github.com/vllm-project/vllm/pull/54474) | The RFC was closed in favor of the config-object direction; the PR also wired the previously missed MoE dispatch-policy variable into environment refresh. | Useful for reproducible M3 experiments: record the resolved config instead of relying on a collection of shell variables. | No |
| Register the model/kernel-specific warmups and graph-safe metadata state needed for sparse MTP replay. | [#53943](https://github.com/vllm-project/vllm/issues/53943) follow-up | The merged ROCm implementation fixed kpool argument/tail propagation, MHC AITER dispatch, and persistent circular-slot metadata; later testing reported CUDA-graph faults gone and 659→2203 tok/s on the tested GLM setup. | Not an M3 patch, but the lesson is direct: graph replay requires persistent slot metadata and complete argument propagation, not merely `--enforce-eager`. | No |
| Register custom operations and fake implementations when a Triton kernel must remain visible to compilation. | [#49699](https://github.com/vllm-project/vllm/issues/49699), [PR #51453](https://github.com/vllm-project/vllm/pull/51453) | Corrected a compile-mode-dependent W4A16 regression. | Audit M3’s custom-op registrations whenever a kernel appears in eager mode but disappears under compilation. | No |

## Actions deliberately **not** counted as successful

These issue threads contain useful constraints, but not a successful optimization to copy:

- **#55132:** the naive smaller workspace reservation increased KV capacity, but graph replay then used freed workspace and faulted. Use the merged bounded-workspace implementation instead.
- **#55351:** forcing the FP32 router weight made the advertised kernel reachable, but throughput was within noise and memory increased; do not enable it blindly.
- **#55687, #55720, #55722:** the large slowdowns were retracted or explained by cache/harness artifacts; the flags are not established regressions from these threads.
- **#43801:** the apparent blocking `hipGraphLaunch` result was attributed to profiler-span differences; no confirmed async-scheduling optimization resulted.
- **#54039:** it documents an unresolved ROCm + speculative-decoding async-scheduling risk, not a validated fix.
- **#57228:** DCP is an active RFC, not a completed gfx942 action; the issue comment says AITER FP4/FP8 MLA query replication is still unavailable.

## First M3/gfx942 shortlist

1. Verify the image includes the merged sparse-indexer/cache fixes and the actual `#54682` MiniMax decode path.
2. A/B **skip logits clearing**, **AITER top-k on the installed version**, and **shape-aware KV-split selection**.
3. Inspect M3’s shared-expert gate/FSE dispatch and MTP weight mapping; do not run both shared-expert optimizations simultaneously.
4. Check the fused multi-step metadata hook, then measure speculative depth separately from target verification cost.
5. Confirm QuickReduce+RMSNorm and every intended custom op are actually selected in the compiled graph.

The shortlist is intentionally validation-first: gfx950 constants, GLM-specific sparse MLA, MXFP4 MoE, and NVIDIA-only DCP work are not assumed to transfer to gfx942 or MiniMax-M3.
