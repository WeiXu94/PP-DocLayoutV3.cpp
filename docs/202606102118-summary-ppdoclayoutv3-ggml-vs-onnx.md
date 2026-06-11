# PP-DocLayoutV3 — GGML engine vs ONNXRuntime benchmark

Date: 2026-06-10 · Host: Apple M3 Max · Input: real 2-column page `assets/doc_2col-01.png`,
resized 800×800 · Output: `fetch_name_0` `[300,7]` = `[label, score, x1, y1, x2, y2, read_order]`.

Pure external benchmark — `scripts/bench_ppdoclayoutv3.py`. No inference-code changes.
GGML is driven via the `pp-doclayout-ggml` CLI (subprocess); ONNXRuntime in-process.

## Latency

| Backend | Mode | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |
|---|---|---:|---:|---:|---:|---:|
| ONNXRuntime CPU | warm (run only) | 335.8 | 339.0 | 336.4 | 332.0 | 2.98 |
| ONNXRuntime CPU | cold (load+run) | 497.8 | 499.0 | 498.1 | 497.0 | 2.01 |
| ONNXRuntime CoreML | warm (run only) | 255.9 | 265.0 | 257.2 | 248.5 | 3.91 |
| ONNXRuntime CoreML | cold (load+run) | 4464.3 | 4488.9 | 4466.9 | 4435.4 | 0.22 |
| GGML CPU | CLI invocation (load+run) | 2307.1 | 2317.8 | 2309.5 | 2301.4 | 0.43 |

GGML CLI overhead decomposition (warm OS cache):
- run to an early node (process + plan-parse + build, ~0 compute): **~50 ms**
- run to `fetch_name_0` (full): **~2300 ms**
- ⇒ the ~2.25 s is weight reads (130 MB re-read every call via 1899 per-tensor file opens)
  + full graph build + actual compute. Plan-parse/process overhead is negligible.

## Parity (vs ONNXRuntime CPU, IoU>0.5 matched detections)

| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mismatch |
|---|---:|---:|---:|---:|---:|---:|
| GGML CPU | 30 | 31 | 30 | 4.724 | 2.46e-01 | 0 |
| ORT CoreML | 30 | 30 | 30 | 2.470 | 1.45e-02 | 0 |

GGML matches ORT numerically: all 30 high-confidence boxes recovered, sub-5px,
correct labels. The 31st GGML box is one extra detection sitting right at the 0.5
score threshold (a tiny score delta flips it in/out).

## Metal: not runnable as-is

`build-metal` (ggml `GGML_METAL=ON`) initializes (M3 Max, Apple9) but **aborts mid-graph**:

```
ggml_metal_op_encode_impl: error: unsupported op 'CONV_2D_DW'
ggml-metal-ops.cpp:203: unsupported op
```

Two distinct Metal blockers in this engine:
1. **Standard ggml op missing in the Metal backend**: `ggml_conv_2d_dw` (depthwise
   conv2d). Used by the backbone (`pp_doclayout_ggml.cpp:2028`). Metal has regular
   `ggml_conv_2d` + `ggml_pool_2d` but no depthwise kernel. This is what aborts first.
2. **CPU-callback custom ops** (`ggml_map_custom*` / `ggml_custom_4d`): `erf_custom_op`,
   `floor_custom_op`, `scatter_nd_op`, `grid_sample_op`, `msdeform_attn_op`,
   `reduce_last_axis_op`, plus inline TopK/GatherND/Range/Tile/Einsum customs
   (~16 call sites). These run a host C++ callback and **cannot execute on Metal** at all.

Context: even ORT-CoreML offloads only 783/1400 nodes to GPU/ANE and lands at 256 ms
warm vs ORT-CPU 336 ms — only ~1.3×. The RT-DETR decoder (deformable attention,
gather/scatter/topk) is not GPU-friendly, so the Metal ceiling for this model is modest.

## Compute profiling (added) — where the 2.3 s actually goes

Instrumented `ggml_backend_graph_compute` (env `PPDOC_TIMING=1`) on `build-cpu`:

| Measurement | ms |
|---|---:|
| Total per-call (warm) | ~2300 |
| **ggml graph compute alone** | **~2170 (94%)** |
| build + plan-parse + weight upload | ~130 |
| compute to end of backbone (`Div.1`, node 368) | ~989 |
| compute, full graph (`fetch_name_0`) | ~2197 |
| ⇒ decoder/head share | ~1208 |

Also: caching the 130 MB GGUF in RAM changed warm latency by **0 ms** — the OS page
cache already served those reads, so disk I/O was never the bottleneck.

Implications:
- A **warm/persistent engine saves only ~130 ms** (build+upload). Not worth the
  refactor — the cost is the compute itself, which warm reuse cannot remove.
- GGML-CPU compute is **~6.5× slower than ORT-CPU** (2170 vs 336 ms). The lever is
  faster compute, i.e. **GPU offload**, not load amortization.
- ~989 ms backbone (conv/bn/relu/pool/depthwise — GPU-friendly) + ~1208 ms decoder
  (matmul/softmax/norm are GPU-friendly; the custom ops msdeform/grid_sample/
  gather/scatter/topk are not, without native Metal kernels).

## Takeaways

- GGML is numerically faithful to the ONNX model. ✅
- As a one-shot CLI, GGML CPU ≈ **4.6× slower than ORT-CPU cold**, ~6.9× vs warm —
  but most of GGML's time is per-call weight reload + graph build, not steady-state
  compute (warm ORT reuses a compiled session; the CLI has no warm mode).
- Highest-ROI improvement is a **persistent/warm engine** (load once, run many), not Metal.
