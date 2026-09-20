# Action sweep instructions

At `20260920-vllm-optimization-issues/`, create an `actions/` subdirectory.

For each action listed in [`../ACTIONS.md`](../ACTIONS.md), create one Markdown file containing exactly these sections, in this order:

## 1. Module diagram

Start with a Mermaid diagram of the module at a granularity matching the action.

Highlight the node or nodes that the action focuses on.

For attention, include concrete nodes such as:

- `load_q`
- `load_k`
- `load_v`
- `compute_index_score`
- top-k and sparse-page-table operations when relevant

For MoE, include concrete nodes such as:

- `sort_token`
- `gate/up`
- `down`
- expert dispatch/combine operations when relevant

## 2. Motivation

Provide the motivation in exactly one sentence.

## 3. Before vs. after

Provide only a table describing behavior changes after the modification.

Do not report performance outcomes, measurements, or expected gains in this section.

The table should describe behavioral changes such as:

- communication being removed or changed;
- unnecessary computation being removed;
- launches being fused or eliminated;
- data movement or materialization changing;
- dispatch, scheduling, or backend-selection behavior changing.

## 4. MiniMax-M3 migration steps

Turn this section into an executable migration decision, not a possibility estimate.

If the action is portable, provide **no more than four numbered steps**:

1. Locate the exact MiniMax-M3 vLLM call path (source file, symbol, and relevant config/launch setting) that corresponds to the action.
2. Port the smallest code change, naming the target symbols and preserving the existing backend, dtype, layout, and parallelism contracts.
3. Add the required shape/dtype/backend/architecture guards and retain the existing fallback for unsupported cases.
4. Run the smallest correctness check that compares the old and new paths, then benchmark only after outputs, graph capture/replay, and relevant M3 decode/prefill shapes match.

Every step must be concrete for MiniMax-M3; cite source files and symbols rather than describing a generic optimization. Keep the M3 snapshot (GPU, vLLM version, model path, backend, dtype, and parallelism) next to the steps when it determines applicability.

If migration is impossible, say **Migration: impossible** and provide source-code-based evidence: cite the action's target file/symbol and MiniMax-M3's matching file/symbol, then identify the blocking mismatch (for example, absent kernel/backend support, incompatible architecture guard, dtype or layout contract, unsupported parallelism, or no reachable call path). Do not infer impossibility from conceptual similarity, another model/GPU result, or an absent/unmerged change; if the source does not establish either a reachable path or a blocker, say **Migration: unverified** instead.
