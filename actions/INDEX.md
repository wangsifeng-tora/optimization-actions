# Action index

Mapping from each action in [`../ACTIONS.md`](../ACTIONS.md) to its generated sweep file, in source order.

| # | Action from `ACTIONS.md` | Sweep file |
|---:|---|---|
| 1 | Replace the MiniMax fresh-decode indexer path with request-balanced score tiling, shared K tiles across query rows, and a fused top-k + sparse page-table Triton path; keep a graph-safe guarded fallback. | [`01.md`](./01.md) |
| 2 | Remove full-pool BF16 materialization from sparse decode: read the quantized cache directly in the Triton decode path; use a caller-sized chunk for prefill gather. | [`02.md`](./02.md) |
| 3 | Eliminate per-decode sparse-MLA fill/fill-functor launches in the hot loop. | [`03.md`](./03.md) |
| 4 | Enable uniform-batch CUDA-graph mode for ROCm sparse MLA, including MTP-shaped batches. | [`04.md`](./04.md) |
| 5 | Select decode KV split counts with a work-per-split heuristic instead of a fixed cap. | [`05.md`](./05.md) |
| 6 | Skip redundant sparse-index remapping on non-indexer layers. | [`06.md`](./06.md) |
| 7 | Skip clearing sparse-prefill MQA logits when the consumer is bounded by real sequence lengths. | [`07.md`](./07.md) |
| 8 | Fuse the DSA indexer prologue: K normalization, Q/K RoPE, FP8 quantization, and K-cache write. | [`08.md`](./08.md) |
| 9 | Make gfx950 top-k split count and 1024-thread policy depend on live row length; add a calibrated `topK=2048` policy. | [`09.md`](./09.md) |
| 10 | Stop clearing the full `max_model_len` sparse-indexer logits buffer on every decode step; pass the equivalent of CUDA’s `clean_logits=False` contract. | [`10.md`](./10.md) |
| 11 | Use the AITER top-k kernel at long context once the installed AITER version supports the shape, instead of retaining the old `>64K` fallback. | [`11.md`](./11.md) |
| 12 | Size the GLM sparse-indexer decode workspace from the actual decode bound/capture bound rather than `max_num_batched_tokens`. | [`12.md`](./12.md) |
| 13 | Fuse decode QK-RoPE, Q concatenation, KV concatenation, and KV-cache write through AITER. | [`13.md`](./13.md) |
| 14 | Use `ROCM_AITER_UNIFIED_ATTN` with AITER enabled instead of allowing the generic `ROCM_ATTN` choice on long-context AMD serving. | [`14.md`](./14.md) |
| 15 | Prefer BF16 KV over FP8 KV when the GPU/backend has expensive FP8 conversion and no native FP8 path. | [`15.md`](./15.md) |
| 16 | Use tuned AITER GEMM for the MoE router gate instead of the generic ATen GEMM path. | [`16.md`](./16.md) |
| 17 | Tune block-FP8 fused MoE for low-batch decode. | [`17.md`](./17.md) |
| 18 | Vectorize the FP32 `moe_sum` reduction and support arbitrary top-k. | [`18.md`](./18.md) |
| 19 | Restore the fast MoE reduce-scatter path by removing an additional communication step; add sequence-parallel support where it improves the shape. | [`19.md`](./19.md) |
| 20 | Retune MXFP4 MoE for gfx950. | [`20.md`](./20.md) |
| 21 | Fuse the shared-expert gate’s skinny GEMM, sigmoid, broadcast, and multiply into one Triton kernel with a shape-guarded fallback. | [`21.md`](./21.md) |
| 22 | Enable fused shared experts for block-quantized FP8, wire the main GLM5Next MoE and its MTP block, and derive the weight mapping from the built module’s fusion flag. | [`22.md`](./22.md) |
| 23 | Register the Triton W4A16 GEMM and its fake/custom op so compile mode does not silently lose the intended kernel. | [`23.md`](./23.md) |
| 24 | Add non-contiguous RMSNorm support to the fast kernel path. | [`24.md`](./24.md) |
| 25 | Remove redundant clone/copy operations in GLM/DeepSeek model paths. | [`25.md`](./25.md) |
| 26 | Dequantize MXFP8 weights once after loading when `dot_scaled` cannot consume the original shape, instead of rebuilding BF16 weights every forward. | [`26.md`](./26.md) |
| 27 | Override DFlash2 draft attention from `ROCM_ATTN` to `TRITON_ATTN` when the ROCm prefix-attention path loses acceptance under batching. | [`27.md`](./27.md) |
| 28 | Opt sparse-MLA/indexer metadata builders into fused multi-step draft decode with an empty update hook when positions/lengths are already advanced by the framework. | [`28.md`](./28.md) |
| 29 | Skip the sparse indexer top-k work for short dense-MHA layers in MTP cases. | [`29.md`](./29.md) |
| 30 | Fuse QuickReduce with RMSNorm on ROCm. | [`30.md`](./30.md) |
| 31 | Replace MoE all-reduce with reduce-scatter and remove an extra communication step in the regression fix. | [`31.md`](./31.md) |
| 32 | Use virtual-batch PCP/DCP-style MLA parallelism to reduce replicated KV-cache pressure. | [`32.md`](./32.md) |
| 33 | Add an `AITERConfig` object to `VllmConfig`, preserving existing environment-variable defaults while making AITER toggles explicit and testable. | [`33.md`](./33.md) |
| 34 | Register the model/kernel-specific warmups and graph-safe metadata state needed for sparse MTP replay. | [`34.md`](./34.md) |
| 35 | Register custom operations and fake implementations when a Triton kernel must remain visible to compilation. | [`35.md`](./35.md) |
