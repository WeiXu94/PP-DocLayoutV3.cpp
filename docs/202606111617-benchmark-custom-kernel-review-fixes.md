# PP-DocLayoutV3 Benchmark - Custom-Kernel Review Fixes

Date: 2026-06-11. Code under test: current uncommitted review fixes for explicit custom-kernel tags, Metal reduce `has_simdgroup_reduction` gating, and CPU MSDeform `1.0f / points` parity.

## Commands

```bash
cmake --build build-cpu --target pp-doclayout-ggml -j8
cmake --build build-metal --target pp-doclayout-ggml -j8
python3 scripts/bench_ppdoclayoutv3.py --onnx /Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx --image assets/doc_2col-01.png --plan artifacts/v3_plan.json --weights artifacts/v3_weights.gguf --manifest artifacts/v3_manifest.json --ggml-cpu build-cpu/pp-doclayout-ggml --ggml-metal build-metal/pp-doclayout-ggml --workdir artifacts/bench/review-change-202606111617 --md-out artifacts/bench/review-change-202606111617/report.md
```

## Result

| Backend | Mode | runs | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |
|---|---|---:|---:|---:|---:|---:|---:|
| ONNXRuntime CPU | warm (run only) | 20 | 340.2 | 365.5 | 347.4 | 337.1 | 2.94 |
| ONNXRuntime CPU | cold (load+run) | 10 | 504.4 | 514.6 | 504.5 | 495.5 | 1.98 |
| ONNXRuntime CoreML | warm (run only) | 20 | 267.0 | 279.0 | 268.7 | 255.9 | 3.75 |
| ONNXRuntime CoreML | cold (load+run) | 10 | 4410.3 | 4432.0 | 4411.9 | 4379.3 | 0.23 |
| GGML CPU | CLI invocation (load+run) | 10 | 2009.0 | 2020.6 | 2011.8 | 2003.1 | 0.50 |
| GGML Metal | CLI invocation (load+run) | 10 | 603.8 | 611.8 | 604.7 | 598.3 | 1.66 |

## Parity vs ONNXRuntime CPU

| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mism. |
|---|---:|---:|---:|---:|---:|---:|
| GGML CPU | 30 | 31 | 30 | 4.724 | 2.46e-01 | 0 |
| GGML Metal | 30 | 31 | 30 | 4.658 | 2.46e-01 | 0 |
| CoreML (ORT CoreML) | 30 | 30 | 30 | 2.470 | 1.45e-02 | 0 |

Raw benchmark report: `artifacts/bench/review-change-202606111617/report.md`.

Notes: ONNXRuntime CoreML emitted the usual partial-offload warnings (`783/1400` nodes supported); benchmark completed. No `PPDOC_TIMING=1` compute-only timing was captured in this run, only CLI load+run timing from the benchmark script.
