# Warm Metal PDF Batch Benchmark

Date: 2026-06-11. Input: `assets/test3-2col-image.pdf` (18 pages). Backend: Metal (`MTL0`). CPU benchmark intentionally skipped.

## Result

| Mode | pages | prepare (ms) | p50 run (ms/page) | p90 run (ms/page) | mean run (ms/page) | min (ms) | max (ms) | pages/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GGML Metal warm batch | 18 | 74.2 | 434.3 | 436.6 | 434.9 | 432.5 | 447.1 | 2.30 |

Warm vs single-run parity on `artifacts/bench/input.f32`: 2100 values, max abs err `0.0`, mean abs err `0.0`.

## Outputs

| File | Purpose |
|---|---|
| `artifacts/bench/pdf-warm-metal/report.md` | Generated benchmark report |
| `artifacts/bench/pdf-warm-metal/detections.json` | Page detections and timing JSON |
| `artifacts/bench/pdf-warm-metal/annotated/` | Annotated page PNGs |
| `artifacts/bench/pdf-warm-metal/outputs/` | Raw `[300,7]` f32 outputs per page |

## Code Change

Added reusable prepared-prefix state to `Engine`, C ABI warm-session entry points, `--batch-f32-list` CLI mode, and `scripts/bench_pdf_warm_batch.py` for PDF render, warm batch inference, and report generation.
