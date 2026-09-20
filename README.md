# Optimization Actions

## MiniMax M3 on MI325x

Dashboard for the current MiniMax-M3 FP8 serving path: 8× MI325X/gfx942, TP8/PP1/DP1, EP off, vLLM `0.26.0+rocm723`, AITER dense/sparse attention, Triton indexer, FP8 main KV, shared-expert fusion, INT4 QuickReduce, and DSpark with draft `TRITON_ATTN`.

These are validation tasks, not claims that gfx950 or another model's result transfers directly to MiniMax-M3/gfx942.

Status: **D** = done on MiniMax-M3/MI325X, **N** = not available, **I** = idle/not yet validated.

### Attention

#### Sparse attention, MLA, and indexer

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Replace the MiniMax fresh-decode indexer path with request-balanced score tiling, shared K tiles across query rows, and a fused top-k + sparse page-table Triton path; keep a graph-safe guarded fallback. | [01](actions/01.md) | I |
| [ ] | Remove full-pool BF16 materialization from sparse decode; read the quantized cache directly in Triton and use a caller-sized prefill chunk. | [02](actions/02.md) | I |
| [ ] | Eliminate per-decode sparse-MLA fill/fill-functor launches in the hot loop. | [03](actions/03.md) | I |
| [ ] | Enable uniform-batch CUDA-graph mode for ROCm sparse MLA, including MTP-shaped batches. | [04](actions/04.md) | I |
| [ ] | Select decode KV split counts with a work-per-split heuristic instead of a fixed cap. | [05](actions/05.md) | I |
| [ ] | Skip redundant sparse-index remapping on non-indexer layers. | [06](actions/06.md) | I |
| [ ] | Skip clearing sparse-prefill MQA logits when consumers are bounded by real sequence lengths. | [07](actions/07.md) | I |
| [ ] | Fuse the DSA indexer prologue: K normalization, Q/K RoPE, FP8 quantization, and K-cache write. | [08](actions/08.md) | I |
| [ ] | Make gfx950 top-k split count and 1024-thread policy depend on live row length; add a calibrated `topK=2048` policy. | [09](actions/09.md) | I |
| [ ] | Stop clearing the full `max_model_len` sparse-indexer logits buffer on every decode step; pass the equivalent of CUDA's `clean_logits=False` contract. | [10](actions/10.md) | I |
| [ ] | Use the AITER top-k kernel at long context when the installed AITER version supports the shape instead of retaining the old `>64K` fallback. | [11](actions/11.md) | I |
| [ ] | Size the GLM sparse-indexer decode workspace from the actual decode/capture bound rather than `max_num_batched_tokens`. | [12](actions/12.md) | I |

#### MLA preparation and backend selection

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Fuse decode QK-RoPE, Q concatenation, KV concatenation, and KV-cache write through AITER. | [13](actions/13.md) | I |
| [ ] | Use `ROCM_AITER_UNIFIED_ATTN` with AITER enabled instead of the generic `ROCM_ATTN` choice on long-context AMD serving. | [14](actions/14.md) | I |
| [ ] | Prefer BF16 KV over FP8 KV when the GPU/backend has expensive FP8 conversion and no native FP8 path. | [15](actions/15.md) | I |

### MoE

#### Routed-expert kernels and routing

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Use tuned AITER GEMM for the MoE router gate instead of generic ATen GEMM. | [16](actions/16.md) | I |
| [ ] | Tune block-FP8 fused MoE for low-batch decode. | [17](actions/17.md) | I |
| [ ] | Vectorize the FP32 `moe_sum` reduction and support arbitrary top-k. | [18](actions/18.md) | I |
| [ ] | Restore the fast MoE reduce-scatter path by removing an additional communication step; add sequence-parallel support where the shape benefits. | [19](actions/19.md) | I |
| [ ] | Retune MXFP4 MoE for gfx950. | [20](actions/20.md) | I |

#### Shared experts and expert gates

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Fuse the shared-expert gate's skinny GEMM, sigmoid, broadcast, and multiply into one Triton kernel with a shape-guarded fallback. | [21](actions/21.md) | I |
| [ ] | Enable fused shared experts for block-quantized FP8, wire the main GLM5Next MoE and its MTP block, and derive weight mapping from the built module's fusion flag. | [22](actions/22.md) | I |

### FFN, normalization, and elementwise overhead

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Register the Triton W4A16 GEMM and its fake/custom op so compile mode does not lose the intended kernel. | [23](actions/23.md) | I |
| [ ] | Add non-contiguous RMSNorm support to the fast kernel path. | [24](actions/24.md) | I |
| [ ] | Remove redundant clone/copy operations in GLM/DeepSeek model paths. | [25](actions/25.md) | I |
| [ ] | Dequantize MXFP8 weights once after loading when `dot_scaled` cannot consume the original shape instead of rebuilding BF16 weights every forward. | [26](actions/26.md) | I |

### Speculative decoding

#### Draft attention and multi-step metadata

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Override DFlash2 draft attention from `ROCM_ATTN` to `TRITON_ATTN` when ROCm prefix attention loses acceptance under batching. | [27](actions/27.md) | I |
| [ ] | Opt sparse-MLA/indexer metadata builders into fused multi-step draft decode with an empty update hook when positions/lengths are already advanced. | [28](actions/28.md) | I |
| [ ] | Skip sparse-indexer top-k work for short dense-MHA layers in MTP cases. | [29](actions/29.md) | I |

### Communication and parallelism

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Fuse QuickReduce with RMSNorm on ROCm. | [30](actions/30.md) | I |
| [ ] | Replace MoE all-reduce with reduce-scatter and remove an extra communication step in the regression fix. | [31](actions/31.md) | I |
| [ ] | Use virtual-batch PCP/DCP-style MLA parallelism to reduce replicated KV-cache pressure. | [32](actions/32.md) | I |

### Runtime, graph, and configuration hygiene

| Check | Description | Link | Status |
|---|---|---|:---:|
| [ ] | Add an `AITERConfig` object to `VllmConfig`, preserving environment defaults while making AITER toggles explicit and testable. | [33](actions/33.md) | I |
| [ ] | Register model/kernel-specific warmups and graph-safe metadata state needed for sparse MTP replay. | [34](actions/34.md) | I |
| [ ] | Register custom operations and fake implementations when a Triton kernel must remain visible to compilation. | [35](actions/35.md) | I |

### Sources

Primary vLLM issue threads:

- Attention: [#54681](https://github.com/vllm-project/vllm/issues/54681), [#41962](https://github.com/vllm-project/vllm/issues/41962), [#57406](https://github.com/vllm-project/vllm/issues/57406), [#57230](https://github.com/vllm-project/vllm/issues/57230), [#55327](https://github.com/vllm-project/vllm/issues/55327), [#55373](https://github.com/vllm-project/vllm/issues/55373), [#55615](https://github.com/vllm-project/vllm/issues/55615), [#56945](https://github.com/vllm-project/vllm/issues/56945), [#56992](https://github.com/vllm-project/vllm/issues/56992)
- MoE and model paths: [#46654](https://github.com/vllm-project/vllm/issues/46654), [#43187](https://github.com/vllm-project/vllm/issues/43187), [#54376](https://github.com/vllm-project/vllm/issues/54376)
- FFN and quantization: [#49699](https://github.com/vllm-project/vllm/issues/49699), [#56506](https://github.com/vllm-project/vllm/issues/56506)
- Speculative decoding: [#53323](https://github.com/vllm-project/vllm/issues/53323), [#54369](https://github.com/vllm-project/vllm/issues/54369)
- Communication and runtime: [#57149](https://github.com/vllm-project/vllm/issues/57149), [#53938](https://github.com/vllm-project/vllm/issues/53938), [#53943](https://github.com/vllm-project/vllm/issues/53943)

See [`actions/INDEX.md`](actions/INDEX.md) for the full action-to-file mapping and [`actions/ACTION_SWEEP.md`](actions/ACTION_SWEEP.md) for the per-action document requirements.
