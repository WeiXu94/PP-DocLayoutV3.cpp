# pp-doclayout-ggml-engine

Native GGML runner for PP-DocLayoutV3, extracted from Archon Tauri as a standalone
engine repository.

## Build

Set `GGML_DIR` to a ggml or llama.cpp checkout containing GGML CMake targets:

```bash
cmake -S . -B build -DGGML_DIR=/path/to/llama.cpp -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build --target pp-doclayout-ggml -j4
```

The CMake project builds:

- `pp_doclayout_ggml`: static library with the C++ engine and C FFI.
- `pp-doclayout-ggml`: CLI for prefix-level debugging.

## FFI

`pp_doclayout_ggml_ffi.h` exposes an opaque `PpDocEngine` handle:

- `ppdoc_create` / `ppdoc_destroy`
- `ppdoc_load_manifest`
- `ppdoc_load_weights`
- `ppdoc_run_plan_prefix`
- `ppdoc_last_error`

`ppdoc_run_plan_prefix` accepts image input as an in-memory `float*` tensor and
writes output rows into a caller-provided buffer.
