# vLLM ROCm Optimization Issues Checklist

Search window: **2026-08-21 through 2026-09-20 UTC**, including issues closed during that window.

Scope: ROCm/AMD issues about performance, optimization, throughput, latency, memory efficiency, or accelerator-path selection. Pure correctness, CI, installation, and model-support issues were excluded.

## Open

### Main optimization trackers

- [ ] [#57149](https://github.com/vllm-project/vllm/issues/57149) — Qwen3.8-2.4T-A95B performance optimization on gfx950/MI355X; tracks MoE, FP8 attention, GDN, QuickReduce, and indexer work.
- [ ] [#57230](https://github.com/vllm-project/vllm/issues/57230) — GLM 5.2/5.3 performance optimization on gfx950/MI355X; AITER sparse-indexer, GEMM, MQA-logits, and MTP work.
- [ ] [#56506](https://github.com/vllm-project/vllm/issues/56506) — DeepSeek-V4.1-Flash ROCm performance RFC; focuses on mHC launch reduction, stream overlap, GEMM tuning, MoE selection, and sparse indexer optimization.
- [ ] [#57228](https://github.com/vllm-project/vllm/issues/57228) — ROCm Decode Context Parallelism for DeepSeek-V4; targets KV-memory reduction and long-context throughput.
- [ ] [#57588](https://github.com/vllm-project/vllm/issues/57588) — Add AITER Gluon sparse-MLA for rope-free BF16 on GLM-5.3/gfx950.
- [ ] [#55916](https://github.com/vllm-project/vllm/issues/55916) — RDNA4 FlyDSL all-reduce backend; measured target is 41–45 GB/s on Radeon AI PRO R9700.

### Specific performance problems

- [ ] [#56992](https://github.com/vllm-project/vllm/issues/56992) — FP8 KV cache is slower than BF16 on RDNA3/gfx1100.
- [ ] [#56945](https://github.com/vllm-project/vllm/issues/56945) — AITER-disabled ROCm image shows 1.7–3.7× slower results; comments clarify the default is intentional but recipe guidance is needed.
- [ ] [#54369](https://github.com/vllm-project/vllm/issues/54369) — Sparse-MLA draft metadata rebuild limits useful MTP depth; later comments report the fused path can be enabled.
- [ ] [#54438](https://github.com/vllm-project/vllm/issues/54438) — gfx1100 attention backend ranking selects a slower path; reported up to 10.7× decode difference at long context.
- [ ] [#55615](https://github.com/vllm-project/vllm/issues/55615) — Sparse-MLA top-k falls back to the vLLM kernel above 64K due to AITER limitations.
- [ ] [#55373](https://github.com/vllm-project/vllm/issues/55373) — Sparse indexer clears a full 1M-token logits buffer every decode step; reported 17% throughput overhead.
- [ ] [#55327](https://github.com/vllm-project/vllm/issues/55327) — gfx950 top-k policy misses `index_topk=2048`; tuned policy measured 25–64% kernel-time reduction.
- [ ] [#55132](https://github.com/vllm-project/vllm/issues/55132) — Sparse-MLA workspace reserves 32 GiB, costing roughly 13.6–21% KV capacity; naive fix currently triggers graph-capture faults.
- [ ] [#55351](https://github.com/vllm-project/vllm/issues/55351) — gfx950 FP32 router GEMM optimization is unreachable because the model weight is BF16.
- [ ] [#55720](https://github.com/vllm-project/vllm/issues/55720) — ROCm fusion passes register no patterns on MLA-only models but still add overhead.
- [ ] [#55722](https://github.com/vllm-project/vllm/issues/55722) — ROCm sparse-MLA graph-partitioning behavior; the original claimed 38% regression was later retracted as a cache/harness artifact.
- [ ] [#53323](https://github.com/vllm-project/vllm/issues/53323) — ROCm DFlash2 acceptance collapses with `ROCM_ATTN`; switching to `TRITON_ATTN` more than doubled throughput at concurrency four.
- [ ] [#54039](https://github.com/vllm-project/vllm/issues/54039) — ROCm + MTP async-scheduling default may conflict with ROCm CI guidance.

### Optimization-adjacent enablement

- [ ] [#54376](https://github.com/vllm-project/vllm/issues/54376) — Fused shared experts unavailable for GLM-5.3 ROCm; comment says fixed by PR #53097.
- [ ] [#53943](https://github.com/vllm-project/vllm/issues/53943) — Missing ROCm `gate_score` path blocks GLM-5.3 sparse-indexer support.
- [ ] [#55609](https://github.com/vllm-project/vllm/issues/55609) — ROCm MLA backend coverage exposes missing decode kernels and tolerance issues.

## Closed during the window

- [x] [#54681](https://github.com/vllm-project/vllm/issues/54681) — MiniMax-M3 fresh decode indexer optimization; completed, closed Sep 1.
- [x] [#53938](https://github.com/vllm-project/vllm/issues/53938) — Config-driven AITER auto-enable RFC; completed, closed Aug 29.
- [x] [#43801](https://github.com/vllm-project/vllm/issues/43801) — Async scheduling on AMD; completed, closed Sep 1.
- [x] [#43187](https://github.com/vllm-project/vllm/issues/43187) — Triton fusion for shared-expert gate, benchmarked on MI355X; completed, closed Sep 19.
- [x] [#49699](https://github.com/vllm-project/vllm/issues/49699) — Compile-mode-3 W4A16 performance degradation on ROCm/gfx908; completed, closed Sep 2.
- [x] [#51142](https://github.com/vllm-project/vllm/issues/51142) — Speed up MI300/MI355 quantization CI groups; completed, closed Sep 1.
- [x] [#55687](https://github.com/vllm-project/vllm/issues/55687) — GLM-5.3 ROCm fusion caused an 8× slowdown; closed not planned Sep 7.
- [x] [#41962](https://github.com/vllm-project/vllm/issues/41962) — Full KV-cache dequantization caused DeepSeek-V4 ROCm OOM; closed not planned Sep 13.

## Related general trackers

- [ ] [#57406](https://github.com/vllm-project/vllm/issues/57406) — GLM 5.3 Performance Optimization; general tracker, not ROCm-specific.
- [ ] [#56217](https://github.com/vllm-project/vllm/issues/56217) — DeepSeek-V4.1-Flash kernels integration and optimization tracker; general tracker.
- [x] [#46654](https://github.com/vllm-project/vllm/issues/46654) — GLM 5.2 Performance Optimization; general tracker, closed Sep 16.

## Summary

Current ROCm optimization work is concentrated on **gfx950/MI355X**, especially sparse MLA/indexer, AITER MoE/GEMM selection, mHC launch overhead, KV workspace sizing, and graph-safe fusion.
