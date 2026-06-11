# Plan — Metal support for PP-DocLayoutV3 GGML engine (custom-op patch)

Goal: run the whole model on Metal. Two blocker classes:
1. `GGML_OP_CONV_2D_DW` — standard ggml op, missing only in the Metal backend.
2. CPU-callback custom ops (`GGML_OP_CUSTOM`) — never dispatched by Metal, and pass
   config via host `userdata` pointers a Metal kernel cannot read.

## Custom op inventory (port targets)

| kernel (pp_doclayout_ggml.cpp) | ONNX op | config source | Metal difficulty |
|---|---|---|---|
| `erf_custom_op`, `floor_custom_op` | Erf, Floor | none | trivial |
| `i32_to_f32_op`, `f32_to_i32_op` | Cast | none | trivial |
| `reduce_last_axis_op` (Max/Min/Sum/Mean) | Reduce* | attr | easy |
| `grid_sample_op` | GridSample | none | moderate |
| inline TopK / GatherND / Range / Tile | TopK/GatherND/Range/Tile | shapes | moderate–hard |
| `scatter_nd_op` | ScatterND | `int64 divisor` (userdata) | moderate |
| `msdeform_attn_op` | deformable attn | `MSDeformConfig` (userdata) | hardest |

## Enabling change: scheduler-based compute (incremental harness for native kernels)

`run_plan_prefix_impl` currently uses a single backend + `ggml_backend_graph_compute`,
so one unsupported op hard-aborts. Switch to `ggml_backend_sched` over `[Metal, CPU]`.
Effect: every op Metal *can* do runs on GPU; the rest fall back to CPU automatically.
This lets us port kernels one at a time and keep a runnable, measurable model the whole
way — it is the dev harness for delivering native kernels, not a substitute for them.

## Patch layout (keep ppdoc logic separable)

- `ggml/src/ggml-metal/ggml-metal.metal` — append `kernel_conv_2d_dw_*` and ppdoc kernels
  (or `#include "ggml-metal-ppdoc.metal.in"` kept as a separate source, concatenated by the
  embed step).
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — add `case GGML_OP_CONV_2D_DW` and a single
  `case GGML_OP_CUSTOM` hook → `ggml_metal_op_ppdoc(ctx, idx)` (dispatch by tensor-name
  prefix `ppdoc.<kind>`).
- `ggml/src/ggml-metal/ggml-metal-device.{cpp,m}` — pipeline getters + `supports_op` cases.
- `ggml/src/ggml-metal/ggml-metal-impl.h` — `ggml_metal_kargs_*` structs.

## Engine-side changes required for the custom ops

- Tag each custom-op output with a stable name `ppdoc.<kind>` so Metal can identify it.
- Move configs out of host `userdata` into device-visible data:
  - small configs (reduce kind, scatter divisor) → `op_params` (≤64 B).
  - `MSDeformConfig` (28 ints) → a small `i32` input tensor (extra `src`).

## Phased delivery (each phase validated vs CPU output, then timed)

1. **sched switch** — model runs on Metal+CPU hybrid. Milestone: no abort; baseline timing.
2. **`conv_2d_dw` Metal kernel** — backbone depthwise convs → GPU. (No engine change.)
3. **trivial ops** — erf/floor/cast/reduce on Metal.
4. **grid_sample** on Metal.
5. **gather/scatter/topk/range/tile** on Metal.
6. **msdeform_attn** on Metal (the decoder hot path).

Validation per op: run a plan prefix to a node just after the op on both `build-cpu` and
`build-metal`; require max abs err ~0. End-to-end: re-run `bench_ppdoclayoutv3.py`,
confirm parity table unchanged, record latency per phase.

## Unresolved questions

1. Custom-op dispatch: route `GGML_OP_CUSTOM`-by-name (smaller core patch, recommended) vs
   promote each to a first-class `GGML_OP_*` (cleaner, larger core patch, other backends
   must mark unsupported)?
2. OK to use `ggml_backend_sched` as the incremental harness (touches `run_plan_prefix_impl`)?
3. Scope for now: do all phases, or stop after the high-value ones (conv_2d_dw + msdeform)?
