# MiniMax-M3 optimization actions on MI325X

Validation checklist for the current MiniMax-M3 FP8 serving path: 8× MI325X/gfx942, TP8/PP1/DP1, EP off, vLLM `0.26.0+rocm723`, AITER dense/sparse attention, Triton indexer, FP8 main KV, shared-expert fusion, INT4 QuickReduce, and DSpark with draft `TRITON_ATTN`.

These are validation tasks, not claims that gfx950 or another model's result transfers directly to MiniMax-M3/gfx942.

## Action checklist

- [ ] **01.** Replace the MiniMax fresh-decode indexer path with request-balanced score tiling, shared K tiles across query rows, and a fused top-k + sparse page-table Triton path; keep a graph-safe guarded fallback. ([details](actions/01.md))
- [ ] **02.** Remove full-pool BF16 materialization from sparse decode; read the quantized cache directly in Triton and use a caller-sized prefill chunk. ([details](actions/02.md))
- [ ] **03.** Eliminate per-decode sparse-MLA fill/fill-functor launches in the hot loop. ([details](actions/03.md))
- [ ] **04.** Enable uniform-batch CUDA-graph mode for ROCm sparse MLA, including MTP-shaped batches. ([details](actions/04.md))
- [ ] **05.** Select decode KV split counts with a work-per-split heuristic instead of a fixed cap. ([details](actions/05.md))
- [ ] **06.** Skip redundant sparse-index remapping on non-indexer layers. ([details](actions/06.md))
- [ ] **07.** Skip clearing sparse-prefill MQA logits when consumers are bounded by real sequence lengths. ([details](actions/07.md))
- [ ] **08.** Fuse the DSA indexer prologue: K normalization, Q/K RoPE, FP8 quantization, and K-cache write. ([details](actions/08.md))
- [ ] **09.** Make gfx950 top-k split count and 1024-thread policy depend on live row length; add a calibrated `topK=2048` policy. ([details](actions/09.md))
- [ ] **10.** Stop clearing the full `max_model_len` sparse-indexer logits buffer on every decode step; pass the equivalent of CUDA's `clean_logits=False` contract. ([details](actions/10.md))
- [ ] **11.** Use the AITER top-k kernel at long context when the installed AITER version supports the shape instead of retaining the old `>64K` fallback. ([details](actions/11.md))
- [ ] **12.** Size the GLM sparse-indexer decode workspace from the actual decode/capture bound rather than `max_num_batched_tokens`. ([details](actions/12.md))
- [ ] **13.** Fuse decode QK-RoPE, Q concatenation, KV concatenation, and KV-cache write through AITER. ([details](actions/13.md))
- [ ] **14.** Use `ROCM_AITER_UNIFIED_ATTN` with AITER enabled instead of the generic `ROCM_ATTN` choice on long-context AMD serving. ([details](actions/14.md))
- [ ] **15.** Prefer BF16 KV over FP8 KV when the GPU/backend has expensive FP8 conversion and no native FP8 path. ([details](actions/15.md))
- [ ] **16.** Use tuned AITER GEMM for the MoE router gate instead of generic ATen GEMM. ([details](actions/16.md))
- [ ] **17.** Tune block-FP8 fused MoE for low-batch decode. ([details](actions/17.md))
- [ ] **18.** Vectorize the FP32 `moe_sum` reduction and support arbitrary top-k. ([details](actions/18.md))
- [ ] **19.** Restore the fast MoE reduce-scatter path by removing an additional communication step; add sequence-parallel support where the shape benefits. ([details](actions/19.md))
- [ ] **20.** Retune MXFP4 MoE for gfx950. ([details](actions/20.md))
- [ ] **21.** Fuse the shared-expert gate's skinny GEMM, sigmoid, broadcast, and multiply into one Triton kernel with a shape-guarded fallback. ([details](actions/21.md))
- [ ] **22.** Enable fused shared experts for block-quantized FP8, wire the main GLM5Next MoE and its MTP block, and derive weight mapping from the built module's fusion flag. ([details](actions/22.md))
- [ ] **23.** Register the Triton W4A16 GEMM and its fake/custom op so compile mode does not lose the intended kernel. ([details](actions/23.md))
- [ ] **24.** Add non-contiguous RMSNorm support to the fast kernel path. ([details](actions/24.md))
- [ ] **25.** Remove redundant clone/copy operations in GLM/DeepSeek model paths. ([details](actions/25.md))
- [ ] **26.** Dequantize MXFP8 weights once after loading when `dot_scaled` cannot consume the original shape instead of rebuilding BF16 weights every forward. ([details](actions/26.md))
- [ ] **27.** Override DFlash2 draft attention from `ROCM_ATTN` to `TRITON_ATTN` when ROCm prefix attention loses acceptance under batching. ([details](actions/27.md))
- [ ] **28.** Opt sparse-MLA/indexer metadata builders into fused multi-step draft decode with an empty update hook when positions/lengths are already advanced. ([details](actions/28.md))
- [ ] **29.** Skip sparse-indexer top-k work for short dense-MHA layers in MTP cases. ([details](actions/29.md))
- [ ] **30.** Fuse QuickReduce with RMSNorm on ROCm. ([details](actions/30.md))
- [ ] **31.** Replace MoE all-reduce with reduce-scatter and remove an extra communication step in the regression fix. ([details](actions/31.md))
- [ ] **32.** Use virtual-batch PCP/DCP-style MLA parallelism to reduce replicated KV-cache pressure. ([details](actions/32.md))
- [ ] **33.** Add an `AITERConfig` object to `VllmConfig`, preserving environment defaults while making AITER toggles explicit and testable. ([details](actions/33.md))
- [ ] **34.** Register model/kernel-specific warmups and graph-safe metadata state needed for sparse MTP replay. ([details](actions/34.md))
- [ ] **35.** Register custom operations and fake implementations when a Triton kernel must remain visible to compilation. ([details](actions/35.md))

## Sources

| # | Source(s) |
|---:|---|
| 1 | [Issue #54681](https://github.com/vllm-project/vllm/issues/54681), [PR #54682](https://github.com/vllm-project/vllm/pull/54682) |
| 2 | [Issue #41962](https://github.com/vllm-project/vllm/issues/41962), [PR #41812](https://github.com/vllm-project/vllm/pull/41812) |
| 3 | [PR #44527](https://github.com/vllm-project/vllm/pull/44527), [Issue #57406](https://github.com/vllm-project/vllm/issues/57406) |
| 4 | [PR #45149](https://github.com/vllm-project/vllm/pull/45149), [Issue #57406](https://github.com/vllm-project/vllm/issues/57406) |
| 5 | [PR #46832](https://github.com/vllm-project/vllm/pull/46832), [Issue #57406](https://github.com/vllm-project/vllm/issues/57406) |
| 6 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51309](https://github.com/vllm-project/vllm/pull/51309) |
| 7 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51314](https://github.com/vllm-project/vllm/pull/51314) |
| 8 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [PR #51315](https://github.com/vllm-project/vllm/pull/51315) |
| 9 | [Issue #55327](https://github.com/vllm-project/vllm/issues/55327) |
| 10 | [Issue #55373](https://github.com/vllm-project/vllm/issues/55373) |
| 11 | [Issue #55615](https://github.com/vllm-project/vllm/issues/55615) |
| 12 | [Issue #57406](https://github.com/vllm-project/vllm/issues/57406), [PR #57701](https://github.com/vllm-project/vllm/pull/57701) |
| 13 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [PR #47757](https://github.com/vllm-project/vllm/pull/47757) |
| 14 | [Issue #56945](https://github.com/vllm-project/vllm/issues/56945) |
| 15 | [Issue #56992](https://github.com/vllm-project/vllm/issues/56992) |
| 16 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [PR #50535](https://github.com/vllm-project/vllm/pull/50535) |
| 17 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46642](https://github.com/vllm-project/vllm/pull/46642) |
| 18 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46643](https://github.com/vllm-project/vllm/pull/46643) |
| 19 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46635](https://github.com/vllm-project/vllm/pull/46635), [PR #47070](https://github.com/vllm-project/vllm/pull/47070), [PR #48763](https://github.com/vllm-project/vllm/pull/48763) |
| 20 | [Issue #57230](https://github.com/vllm-project/vllm/issues/57230), [AITER PR #5045](https://github.com/ROCm/aiter/pull/5045) |
| 21 | [Issue #43187](https://github.com/vllm-project/vllm/issues/43187), [PR #43190](https://github.com/vllm-project/vllm/pull/43190) |
| 22 | [Issue #54376](https://github.com/vllm-project/vllm/issues/54376), [PR #53097](https://github.com/vllm-project/vllm/pull/53097) |
| 23 | [Issue #49699](https://github.com/vllm-project/vllm/issues/49699), [PR #51453](https://github.com/vllm-project/vllm/pull/51453) |
| 24 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #49750](https://github.com/vllm-project/vllm/pull/49750) |
| 25 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46651](https://github.com/vllm-project/vllm/pull/46651) |
| 26 | [Issue #56506](https://github.com/vllm-project/vllm/issues/56506), [PR #56560](https://github.com/vllm-project/vllm/pull/56560) |
| 27 | [Issue #53323](https://github.com/vllm-project/vllm/issues/53323) |
| 28 | [Issue #54369](https://github.com/vllm-project/vllm/issues/54369) |
| 29 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #50904](https://github.com/vllm-project/vllm/pull/50904) |
| 30 | [Issue #57149](https://github.com/vllm-project/vllm/issues/57149), [PR #48249](https://github.com/vllm-project/vllm/pull/48249) |
| 31 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46635](https://github.com/vllm-project/vllm/pull/46635), [PR #48763](https://github.com/vllm-project/vllm/pull/48763) |
| 32 | [Issue #46654](https://github.com/vllm-project/vllm/issues/46654), [PR #46570](https://github.com/vllm-project/vllm/pull/46570), [PR #46076](https://github.com/vllm-project/vllm/pull/46076) |
| 33 | [Issue #53938](https://github.com/vllm-project/vllm/issues/53938), [PR #54474](https://github.com/vllm-project/vllm/pull/54474) |
| 34 | [Issue #53943](https://github.com/vllm-project/vllm/issues/53943) follow-up |
| 35 | [Issue #49699](https://github.com/vllm-project/vllm/issues/49699), [PR #51453](https://github.com/vllm-project/vllm/pull/51453) |

See [`actions/INDEX.md`](actions/INDEX.md) for the full action-to-file mapping and [`actions/ACTION_SWEEP.md`](actions/ACTION_SWEEP.md) for the per-action document requirements.
