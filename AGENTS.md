# Agent Notes

## Project Shape
- Root project is a C++17 CMake wrapper for a PP-DocLayoutV3 GGML runner; main logic is `pp_doclayout_ggml.cpp`, CLI is `main.cpp`, C ABI is `pp_doclayout_ggml_ffi.*`.
- `ggml/` is a submodule but also carries local backend/custom-kernel work; inspect it when touching Metal scheduling or custom ops.
- `artifacts/` is gitignored but expected by local runs: `v3_manifest.json`, `v3_plan.json`, `v3_weights.gguf`, and `bench/input.f32`.

## Build
- CMake requires `GGML_DIR`; use the local submodule unless testing another checkout.
- CPU build: `cmake -S . -B build-cpu -DGGML_DIR=ggml -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu --target pp-doclayout-ggml -j8`.
- Metal build: `cmake -S . -B build-metal -DGGML_DIR=ggml -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-metal --target pp-doclayout-ggml -j8`.
- Root builds add `ggml` with tests/examples off; `ctest` usually reports no tests. Use build plus CLI/benchmark runs for verification.

## Run And Verify
- Full prefix run: `PPDOC_TIMING=1 ./build-metal/pp-doclayout-ggml --manifest artifacts/v3_manifest.json --weights artifacts/v3_weights.gguf --plan artifacts/v3_plan.json --run-prefix fetch_name_0 --input-f32 artifacts/bench/input.f32 --output-f32 /tmp/o.f32`.
- `PPDOC_TIMING=1` prints `graph_compute_ms`; `GGML_SCHED_DEBUG=2` prints backend assignment/splits.
- Benchmark/parity: `python3 scripts/bench_ppdoclayoutv3.py --onnx <PP-DocLayoutV3.onnx> --image assets/doc_2col-01.png --plan artifacts/v3_plan.json --weights artifacts/v3_weights.gguf --manifest artifacts/v3_manifest.json --ggml-cpu build-cpu/pp-doclayout-ggml --ggml-metal build-metal/pp-doclayout-ggml`.
- When running a new benchmark, append the result table, parity table, and brief code-change note to `docs/benckmarks.md`.
- Deterministic prefix references come from `scripts/pp_doclayoutv3_prefix_reference.py`; it writes the matching `--input-f32` for `--run-prefix <onnx_output_name>` comparisons.
- Model input is fixed at `1x3x800x800` RGB NCHW float32 with values scaled to `[0,1]`; do not apply ImageNet mean/std.

## Gotchas
- `--detect` is still gated; use `--run-prefix fetch_name_0` for end-to-end detection output `[300,7]`.
- The engine picks preferred backend in order iGPU/GPU/ACCEL/CPU; Metal builds run a preferred+CPU scheduler, CPU builds avoid Metal by configure flags.
- Native Metal custom ops use explicit tags from `ggml/include/ggml-custom-kernels.h`; every tagged op must keep a CPU callback fallback.
- Latest Metal/custom-op context is in `docs/202606111541-summary-metal-custom-decoder-ops.md`.
