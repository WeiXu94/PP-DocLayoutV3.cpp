# PP-DocLayoutV3 GGML — custom decoder ops on Metal (4.1× compute over CPU)

Date: 2026-06-11 · Apple M3 Max · real 2-col page, 800×800, output `fetch_name_0 [300,7]`.
Continues `202606102350-summary-metal-acceleration-results.md` (which ended at 887 ms compute / 1046 ms CLI).

## Result

| Backend | Mode | p50 (ms) | vs GGML-CPU |
|---|---|---:|---:|
| ONNXRuntime CoreML | warm (run only) | 259 | — |
| ONNXRuntime CPU | warm (run only) | 335 | — |
| ONNXRuntime CPU | cold (load+run) | 496 | — |
| **GGML Metal** | CLI (load+run) | **615** | **3.3× faster** |
| GGML CPU | CLI (load+run) | 2000 | 1.0× |

Pure `ggml_backend_graph_compute`: **CPU ~1890 ms → Metal ~458 ms (4.1×)**; was 887 ms
at the start of this pass. Graph splits: 238 → **3** (only the final tiny
`index_put` scatter stays on CPU). Parity unchanged: 30/30 ORT-CPU detections
matched (IoU>0.5), max box err 4.66 px, 0 label mismatches — identical to the
GGML-CPU build, i.e. Metal adds no error of its own (Metal-vs-CPU build direct
match: 31/31, max box err 0.066 px). CPU build stays bit-exact vs. the original.

## What made it work (3 changes, in impact order)

### 1. Weights buffer on the Metal buffer type (engine, `run_plan_prefix_impl`)
Before: weights lived in a CPU host buffer tagged `USAGE_WEIGHTS` and relied on
`op_offload` — but `offload_op` only accepts matmul-like ops, so every
elementwise op that reads a per-channel BN constant got **pinned to CPU**,
splitting the graph at every conv (238 splits, ping-pong copies).
Fix: allocate the wctx tensors from `ggml_backend_get_default_buffer_type(backend)`
(the Metal buffer when Metal is active). The scheduler then assigns every
weight-consuming op to Metal directly. 238 → 47 splits, 887 → 758 ms.
CPU-only build unaffected (same buffer as before).

### 2. Native ggml ops replace custom CPU callbacks (engine)
- `Floor`/`Mod` → `ggml_floor` (GGML_UNARY_OP_FLOOR, Metal-supported).
- F32↔I32 casts (gather indices, Cast nodes) → `ggml_cast` (Metal CPY supports
  F32↔I32; CPU `dup` semantics match the old custom truncation). I64→F32 keeps
  the custom callback (rare, tiny).
758 → 591 ms, 47 → 31 splits.

### 3. Tagged GGML_OP_CUSTOM ops run natively on Metal (ggml fork + engine)
New convention in `ggml/include/ggml-custom-kernels.h`: a custom op whose
userdata starts with `ggml_custom_kernel_hdr { magic='GMCK', kind }` can be
recognized and executed natively by the Metal backend; unaware backends just run
the CPU callback as before. Kinds implemented:
- `ERF` — elementwise erf (`erf_approx`, the GELU_ERF polynomial, ~1.5e-7 abs err).
- `REDUCE_MAX/MIN/SUM` — row-reduce over ne0, one threadgroup per row with
  simdgroup reduction. This was the hidden whale: 5 ReduceMax/Min nodes each
  pulled a **45 MB** mask tensor to CPU (~225 MB of GPU→CPU traffic per run).
- `MSDEFORM_ATTN` — the fused RT-DETR deformable attention (6 decoder layers,
  12 MB value tensor each): one simdgroup per (head, query), lane = head_dim
  channel; same sampling math as the CPU reference.
ggml fork files: `ggml-impl.h` (`ggml_custom_kernel_hdr_from_op` helper),
`ggml-metal-impl.h` (kargs), `ggml-metal.metal` (3 kernels),
`ggml-metal-device.{h,cpp}` (pipelines), `ggml-metal-device.m` (`supports_op`),
`ggml-metal-ops.{h,cpp}` (`ggml_metal_op_custom` dispatch).
591 → 458 ms, 31 → 3 splits.

## Remaining gap vs ORT
Metal compute 458 ms vs ORT-CoreML warm 259 ms / ORT-CPU warm 335 ms. The
remaining difference is per-kernel efficiency of the dense conv/matmul path
(ggml Metal conv via im2col+matmul vs CoreML's fused kernels), no longer
scheduling or custom-op overhead. CLI numbers also carry ~150 ms of process
start + load + graph build per invocation.

## Repro
```
cmake --build build-metal -j8 && cmake --build build-cpu -j8
PPDOC_TIMING=1 ./build-metal/pp-doclayout-ggml --manifest artifacts/v3_manifest.json \
  --weights artifacts/v3_weights.gguf --plan artifacts/v3_plan.json \
  --run-prefix fetch_name_0 --input-f32 artifacts/bench/input.f32 --output-f32 /tmp/o.f32
python3 scripts/bench_ppdoclayoutv3.py --onnx ~/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --image assets/doc_2col-01.png --plan artifacts/v3_plan.json --weights artifacts/v3_weights.gguf \
  --manifest artifacts/v3_manifest.json --ggml-cpu build-cpu/pp-doclayout-ggml --ggml-metal build-metal/pp-doclayout-ggml
```
