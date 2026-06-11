# Warm Metal PDF Batch Benchmark

Date: 2026-06-11. Input: `assets/test3-2col-image.pdf` (18 pages). Backends: GGML Metal (`MTL0`), ONNXRuntime CPU, ONNXRuntime CoreML. GGML-CPU benchmark intentionally skipped.

## Result

| Backend | Mode | pages | prepare/session (ms) | p50 run (ms/page) | p90 run (ms/page) | mean run (ms/page) | min (ms) | max (ms) | pages/s p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GGML Metal | warm batch | 18 | 74.1 | 435.0 | 435.9 | 436.3 | 434.0 | 458.6 | 2.30 |
| ONNXRuntime CPU | warm session | 18 | 210.7 | 339.2 | 343.3 | 342.7 | 337.3 | 370.1 | 2.95 |
| ONNXRuntime CoreML | warm session | 18 | 4271.2 | 299.4 | 312.7 | 305.4 | 269.8 | 467.3 | 3.34 |

Warm vs single-run parity on `artifacts/bench/input.f32`: 2100 values, max abs err `0.0`, mean abs err `0.0`.

## Outputs

| File | Purpose |
|---|---|
| `artifacts/bench/pdf-warm-metal/report.md` | Generated benchmark report |
| `artifacts/bench/pdf-warm-metal/detections.json` | Page detections and timing JSON |
| `artifacts/bench/pdf-warm-metal/annotated/` | Annotated page PNGs |
| `artifacts/bench/pdf-warm-metal/outputs/` | Raw `[300,7]` f32 outputs per page |

## Code Change

Added reusable prepared-prefix state to `Engine`, C ABI warm-session entry points, `--batch-f32-list` CLI mode, and `scripts/bench_pdf_warm_batch.py` for PDF render, GGML Metal warm batch inference, ORT CPU/CoreML warm-session timing, and report generation.
