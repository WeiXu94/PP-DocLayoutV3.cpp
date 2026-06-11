# PP-DocLayoutV3 Merged Benchmark Results

This file merges benchmark records found in `docs/` and `artifacts/bench/`. Each run section is keyed by the timestamp of the source record. Intermediate prefix/gate parity tables are excluded unless they came from the end-to-end benchmark path.

## Summary

| Run | Main change | GGML CPU p50 | GGML Metal p50 | Compute-only note | Parity vs ORT CPU |
|---|---|---:|---:|---|---|
| 202606102118 | Baseline external benchmark; no inference-code change; Metal build aborted on missing `CONV_2D_DW`. | 2307.1 ms | n/a | CPU graph compute ~2170 ms | GGML CPU 30/30, 0 label mismatches |
| 202606102350 | Metal+CPU scheduler, separate `USAGE_WEIGHTS` buffer, `conv_2d_dw` Metal kernel, scheduler/offload ggml patches. | 2005 ms | 1046 ms | CPU ~1886 ms, Metal ~887 ms | 30/30, max box err 4.73 px, 0 label mismatches |
| 202606111541 | Custom decoder ops moved to Metal: Metal weight buffer, native Floor/Mod/Cast, tagged ERF/Reduce/MSDeform custom kernels. | 1999.5 ms | 615.0 ms | CPU ~1890 ms, Metal ~458 ms | GGML CPU 30/30, GGML Metal 30/30, 0 label mismatches |
| 202606111617 | Code-review fixes: explicit custom-kernel op tags, reduce `has_simdgroup_reduction` gate, CPU MSDeform `1.0f / points`. | 2009.0 ms | 603.8 ms | Not captured | GGML CPU 30/30, GGML Metal 30/30, 0 label mismatches |
| 202606111706 | Warm prepared-prefix session plus Metal batch CLI/PDF driver over `assets/test3-2col-image.pdf` (18 pages), compared with ORT CPU/CoreML warm sessions. | not run | 435.0 ms/page warm | ORT CPU 339.2 ms/page, ORT CoreML 299.4 ms/page; GGML prepare 74.1 ms | Warm batch vs single Metal output exact, max abs err 0 |

## 202606102118 - Baseline GGML CPU vs ONNXRuntime

Source: `docs/202606102118-summary-ppdoclayoutv3-ggml-vs-onnx.md`.

What changed: external benchmark only; no inference-code changes. Metal was not runnable because `GGML_OP_CONV_2D_DW` was unsupported in Metal and CPU-callback custom ops could not execute there.

| Backend | Mode | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |
|---|---|---:|---:|---:|---:|---:|
| ONNXRuntime CPU | warm (run only) | 335.8 | 339.0 | 336.4 | 332.0 | 2.98 |
| ONNXRuntime CPU | cold (load+run) | 497.8 | 499.0 | 498.1 | 497.0 | 2.01 |
| ONNXRuntime CoreML | warm (run only) | 255.9 | 265.0 | 257.2 | 248.5 | 3.91 |
| ONNXRuntime CoreML | cold (load+run) | 4464.3 | 4488.9 | 4466.9 | 4435.4 | 0.22 |
| GGML CPU | CLI invocation (load+run) | 2307.1 | 2317.8 | 2309.5 | 2301.4 | 0.43 |

| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mismatch |
|---|---:|---:|---:|---:|---:|---:|
| GGML CPU | 30 | 31 | 30 | 4.724 | 2.46e-01 | 0 |
| ORT CoreML | 30 | 30 | 30 | 2.470 | 1.45e-02 | 0 |

Compute profiling: full CPU graph compute was ~2170 ms of ~2300 ms total. Build + plan parse + weight upload was ~130 ms.

## 202606102350 - First Metal Hybrid Acceleration

Source: `docs/202606102350-summary-metal-acceleration-results.md`.

What changed: `run_plan_prefix_impl` switched to `ggml_backend_sched` over Metal+CPU with `op_offload=true`; weights were placed in a separate host buffer tagged `USAGE_WEIGHTS`; ggml fork gained scheduler view-source offload fix, `conv_2d_dw` Metal kernel, and `CONV_2D_DW` offload support.

| Backend | Mode | p50 (ms) | vs GGML-CPU |
|---|---|---:|---:|
| ONNXRuntime CoreML | warm (run only) | 267 | n/a |
| ONNXRuntime CPU | warm (run only) | 336 | n/a |
| ONNXRuntime CPU | cold (load+run) | 512 | n/a |
| GGML Metal-hybrid | CLI (load+run) | 1046 | 1.9x faster |
| GGML CPU | CLI (load+run) | 2005 | 1.0x |

Compute profiling: `PPDOC_TIMING=1` reported CPU ~1886 ms to Metal ~887 ms. Scheduler assigned 305 ops to Metal and 3259 to CPU. Detection parity was preserved: 30/30 ORT-CPU detections matched, max box err 4.73 px, 0 label mismatches.

## 202606111541 - Custom Decoder Ops On Metal

Sources: `docs/202606111541-summary-metal-custom-decoder-ops.md`, `artifacts/bench/report.md`.

What changed: weights were allocated on the preferred backend buffer type so BN and other weight-consuming elementwise ops could stay on Metal; Floor/Mod and F32/I32 casts moved to native ggml ops; tagged custom Metal kernels were added for ERF, row ReduceMax/Min/Sum, and fused MSDeformAttn. `artifacts/bench/report.md` is the first raw benchmark report and gives the detailed table below.

| Backend | Mode | runs | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |
|---|---|---:|---:|---:|---:|---:|---:|
| ONNXRuntime CPU | warm (run only) | 20 | 335.3 | 336.9 | 335.4 | 334.1 | 2.98 |
| ONNXRuntime CPU | cold (load+run) | 10 | 495.7 | 501.6 | 496.2 | 492.5 | 2.02 |
| ONNXRuntime CoreML | warm (run only) | 20 | 258.7 | 274.5 | 261.7 | 249.2 | 3.87 |
| ONNXRuntime CoreML | cold (load+run) | 10 | 4449.8 | 4628.5 | 4484.7 | 4314.0 | 0.22 |
| GGML CPU | CLI invocation (load+run) | 10 | 1999.5 | 2013.5 | 2002.8 | 1988.9 | 0.50 |
| GGML Metal | CLI invocation (load+run) | 10 | 615.0 | 641.7 | 615.8 | 582.3 | 1.63 |

| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mism. |
|---|---:|---:|---:|---:|---:|---:|
| GGML CPU | 30 | 31 | 30 | 4.724 | 2.46e-01 | 0 |
| GGML Metal | 30 | 31 | 30 | 4.658 | 2.46e-01 | 0 |
| CoreML (ORT CoreML) | 30 | 30 | 30 | 2.470 | 1.45e-02 | 0 |

Compute profiling from the docs summary: CPU ~1890 ms to Metal ~458 ms. Graph splits dropped from 238 to 3; only final tiny `index_put` scatter stayed on CPU.

## 202606111617 - Code Review Fixes Benchmark

Sources: `docs/202606111617-benchmark-custom-kernel-review-fixes.md`, `artifacts/bench/review-change-202606111617/report.md`.

What changed: review fixes made custom-kernel tagging explicit in `op_params` before Metal probes userdata, gated custom reduce kernels on `has_simdgroup_reduction`, and changed the CPU MSDeform fallback from hardcoded `0.25f` to `1.0f / points`.

| Backend | Mode | runs | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |
|---|---|---:|---:|---:|---:|---:|---:|
| ONNXRuntime CPU | warm (run only) | 20 | 340.2 | 365.5 | 347.4 | 337.1 | 2.94 |
| ONNXRuntime CPU | cold (load+run) | 10 | 504.4 | 514.6 | 504.5 | 495.5 | 1.98 |
| ONNXRuntime CoreML | warm (run only) | 20 | 267.0 | 279.0 | 268.7 | 255.9 | 3.75 |
| ONNXRuntime CoreML | cold (load+run) | 10 | 4410.3 | 4432.0 | 4411.9 | 4379.3 | 0.23 |
| GGML CPU | CLI invocation (load+run) | 10 | 2009.0 | 2020.6 | 2011.8 | 2003.1 | 0.50 |
| GGML Metal | CLI invocation (load+run) | 10 | 603.8 | 611.8 | 604.7 | 598.3 | 1.66 |

| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mism. |
|---|---:|---:|---:|---:|---:|---:|
| GGML CPU | 30 | 31 | 30 | 4.724 | 2.46e-01 | 0 |
| GGML Metal | 30 | 31 | 30 | 4.658 | 2.46e-01 | 0 |
| CoreML (ORT CoreML) | 30 | 30 | 30 | 2.470 | 1.45e-02 | 0 |

No `PPDOC_TIMING=1` compute-only timing was captured in this run; this is CLI load+run timing from `scripts/bench_ppdoclayoutv3.py`.

## 202606111706 - Warm Metal PDF Batch

Source: `artifacts/bench/pdf-warm-metal/report.md`, `artifacts/bench/pdf-warm-metal/detections.json`.

What changed: added a reusable prepared-prefix session in the C++ engine, C ABI prepare/run/clear entry points, `--batch-f32-list` CLI mode, and `scripts/bench_pdf_warm_batch.py` for PDF page rendering, GGML Metal warm batch inference, ORT CPU/CoreML warm-session timing, detection JSON, and annotated page outputs. GGML-CPU benchmark was intentionally skipped by request.

| Backend | Mode | pages | prepare/session (ms) | p50 run (ms/page) | p90 run (ms/page) | mean run (ms/page) | min (ms) | max (ms) | pages/s p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GGML Metal | warm batch | 18 | 74.1 | 435.0 | 435.9 | 436.3 | 434.0 | 458.6 | 2.30 |
| ONNXRuntime CPU | warm session | 18 | 210.7 | 339.2 | 343.3 | 342.7 | 337.3 | 370.1 | 2.95 |
| ONNXRuntime CoreML | warm session | 18 | 4271.2 | 299.4 | 312.7 | 305.4 | 269.8 | 467.3 | 3.34 |

| Page | detections >= 0.5 | run (ms) | top labels |
|---:|---:|---:|---|
| 1 | 29 | 458.6 | text=11, header=5, footnote=4, paragraph_title=3 |
| 2 | 14 | 436.6 | text=9, footnote=3, number=1, header=1 |
| 3 | 17 | 434.7 | text=9, footnote=4, paragraph_title=2, header=1 |
| 4 | 22 | 434.9 | footnote=7, text=6, paragraph_title=3, inline_formula=2 |
| 5 | 20 | 435.9 | text=8, footnote=4, paragraph_title=3, header=1 |
| 6 | 19 | 434.4 | text=6, chart=5, footnote=3, figure_title=2 |
| 7 | 25 | 436.6 | text=11, footnote=8, header=1, number=1 |
| 8 | 17 | 434.2 | text=4, figure_title=3, footnote=3, chart=2 |
| 9 | 32 | 434.0 | text=12, footnote=8, inline_formula=4, paragraph_title=4 |
| 10 | 19 | 434.7 | footnote=6, text=5, figure_title=2, number=1 |
| 11 | 22 | 435.3 | text=12, paragraph_title=3, footnote=3, header=1 |
| 12 | 33 | 435.1 | inline_formula=11, text=8, figure_title=3, display_formula=3 |
| 13 | 64 | 434.8 | inline_formula=28, text=16, display_formula=9, formula_number=6 |
| 14 | 41 | 435.2 | inline_formula=15, text=12, display_formula=5, formula_number=4 |
| 15 | 30 | 434.6 | text=14, footnote=4, paragraph_title=3, inline_formula=3 |
| 16 | 22 | 435.1 | text=7, footnote=3, chart=2, figure_title=2 |
| 17 | 17 | 434.9 | text=8, paragraph_title=3, chart=2, figure_title=2 |
| 18 | 57 | 435.5 | reference_content=52, reference=2, number=1, header=1 |

| Comparison | values | max abs err | mean abs err |
|---|---:|---:|---:|
| Metal single CLI vs Metal warm batch on `artifacts/bench/input.f32` | 2100 | 0.0 | 0.0 |

Warm batch detection outputs are in `artifacts/bench/pdf-warm-metal/outputs/`, page-level JSON is in `artifacts/bench/pdf-warm-metal/detections.json`, and annotated page PNGs are in `artifacts/bench/pdf-warm-metal/annotated/`.
