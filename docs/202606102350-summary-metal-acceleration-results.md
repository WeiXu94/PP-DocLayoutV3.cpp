# PP-DocLayoutV3 GGML — Metal acceleration achieved

Date: 2026-06-10 · Apple M3 Max · real 2-col page, 800×800, output `fetch_name_0 [300,7]`.

## Result

| Backend | Mode | p50 (ms) | vs GGML-CPU |
|---|---|---:|---:|
| ONNXRuntime CoreML | warm (run only) | 267 | — |
| ONNXRuntime CPU | warm (run only) | 336 | — |
| ONNXRuntime CPU | cold (load+run) | 512 | — |
| **GGML Metal-hybrid** | CLI (load+run) | **1046** | **1.9× faster** |
| GGML CPU | CLI (load+run) | 2005 | 1.0× |

Pure `ggml_backend_graph_compute` time (PPDOC_TIMING=1): **CPU ~1886 ms → Metal ~887 ms (2.1×)**.
Scheduler assigns **305 ops to Metal**, 3259 to CPU (the custom decoder ops + small
elementwise stay on CPU). Detection parity preserved: 30/30 ORT-CPU detections matched,
max box err 4.73 px, 0 label mismatches (same as GGML-CPU).

## What made it work

### Main repo (`pp_doclayout_ggml.{hpp,cpp}`) — `run_plan_prefix_impl`
1. `ggml_backend_sched` over `[Metal, CPU]` (`op_offload=true`) instead of single-backend
   compute — runs each op on Metal when supported, CPU otherwise.
2. **Weights-buffer split**: constant/weight leaves are built in a separate `wctx` and
   allocated up front via `ggml_backend_alloc_ctx_tensors_from_buft(host_buft)` +
   `ggml_backend_buffer_set_usage(USAGE_WEIGHTS)`. Runtime inputs (image/im_shape/
   scale_factor) stay in the compute ctx and are `ggml_set_input`. This is the trigger
   for the scheduler's GPU offload.
3. RAM weight cache + `PPDOC_TIMING=1` compute timer (gated).

### ggml fork — 3 patches (replay these on your ggml fork)
1. **`ggml-backend.cpp` `ggml_backend_sched_backend_id_from_cur`** (the key bug fix):
   the weight-placement/offload check only looked at the direct src buffer, missing
   weights that reach the op through a reshape *view* (e.g. `ggml_conv_2d` reshapes its
   kernel → the mat-mul's weight operand is an unallocated view at split time). Fix:
   follow `view_src` to the underlying tensor before the `USAGE_WEIGHTS` check.
   → Without this, **0 ops** offloaded; with it, **270**.
2. **`conv_2d_dw` Metal kernel** (depthwise conv2d, missing in ggml-metal): added across
   `ggml-metal-impl.h` (kargs), `ggml-metal.metal` (`kernel_conv_2d_dw_f32`),
   `ggml-metal-device.{cpp,h}` (pipeline), `ggml-metal-device.m` (`supports_op`),
   `ggml-metal-ops.{h,cpp}` (dispatch + encode). Validated ~1e-4 vs CPU.
3. **`ggml-metal.cpp` `offload_op`**: also return true for `GGML_OP_CONV_2D_DW` so
   depthwise convs offload. → 270 → 305 Metal ops, ~950 → ~887 ms.

## Headroom

Remaining ~887 ms is dominated by the custom decoder ops on CPU (msdeform/grid_sample/
gather/scatter/topk) + GPU↔CPU copies around them. Native Metal kernels for those (the
big Path-2 effort) would push further, but the ceiling is bounded — ORT-CoreML, fully
GPU/ANE, is only ~267 ms and offloads just 783/1400 nodes. GGML-CPU's own kernels are
the deeper gap (~6.5× ORT-CPU even before GPU).

## Repro
```
cmake -S . -B build-metal -DGGML_DIR=ggml -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-metal -j8
PPDOC_TIMING=1 ./build-metal/pp-doclayout-ggml --manifest artifacts/v3_manifest.json \
  --weights artifacts/v3_weights.gguf --plan artifacts/v3_plan.json \
  --run-prefix fetch_name_0 --input-f32 artifacts/bench/input.f32 --output-f32 /tmp/o.f32
python3 scripts/bench_ppdoclayoutv3.py --onnx ~/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --image assets/doc_2col-01.png --plan artifacts/v3_plan.json --weights artifacts/v3_weights.gguf \
  --manifest artifacts/v3_manifest.json --ggml-cpu build-cpu/pp-doclayout-ggml --ggml-metal build-metal/pp-doclayout-ggml
```
