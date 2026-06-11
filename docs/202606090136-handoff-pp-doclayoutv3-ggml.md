# Handoff: PP-DocLayoutV3 GGML Engine

## Goal

Continue implementing local PP-DocLayoutV3 inference with GGML/C++ for the Tauri Archon app, replacing the current ONNXRuntime direction only after full detector parity is proven.

Do not mark the goal complete yet. Full `--detect` is still intentionally gated.

## Current State

Authoritative implementation/status note:

- `docs/202606081640-pp-doclayoutv3-ggml-engine.md`

Current worktree contains uncommitted/untracked progress:

- Rust layout contract: `crates/archon-core/src/layout.rs`
- Rust exports/status/CLI hook: `crates/archon-core/src/lib.rs`, `crates/archon-core/src/models.rs`, `crates/archon-cli/src/main.rs`
- Native engine scaffold: `native/pp-doclayout-ggml/`
- Conversion/reference scripts: `scripts/pp_doclayoutv3_onnx_to_ggml_manifest.py`, `scripts/pp_doclayoutv3_prefix_reference.py`
- `.gitignore` now ignores Python cache files.

Native GGML currently verifies:

- `--first-block` through `p2o.pd_op.relu.0.0`
- `--stem-block` through `p2o.pd_op.relu.3.0`
- Both match ONNXRuntime within roughly `1e-6` max absolute error.

## Important Local Paths

- ONNX model: `/Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx`
- Config: `/Users/weixu/models/pp-doclayoutv3-onnx/config.json`
- GGML checkout used for native build: `/tmp/badlogic-ggml-inspect`
- Python 3.12 dependency overlay: `/tmp/onnx-inspect-py312`
- Generated artifacts: `target/native/pp-doclayoutv3.manifest.json`, `target/native/pp-doclayoutv3.plan.json`, `target/native/pp-doclayoutv3-f32.gguf`

## Key Decisions Already Made

- Preprocess must follow model config: resize `800x800`, `keep_ratio=false`, `NCHW`, mean `[0,0,0]`, std `[1,1,1]`.
- Do not use ImageNet normalization.
- Use staged, verified graph slices against ONNXRuntime intermediates.
- Represent ONNX `SAME_UPPER` even-kernel padding with `ggml_pad_ext` before GGML conv/pool ops.
- Keep current app ingestion path unchanged until `--detect` matches ONNX reference labels/boxes.

## Next Work

1. Generalize the hard-coded stem runner into a small generated executor over `target/native/pp-doclayoutv3.plan.json`.
2. Add depthwise conv support starting at node 62. GGML has `ggml_conv_2d_dw` and `ggml_conv_2d_dw_direct`; verify tensor layout against ONNXRuntime before continuing.
3. Continue convolutional backbone block by block, adding intermediate ONNX reference outputs as gates.
4. After backbone parity, implement remaining graph families in priority order: shape/view ops, matmul/add/mul/activations, attention/layernorm, detector postprocess.
5. Add Metal/Vulkan build variants only after CPU correctness is stable.

## Verification Commands

Use the verified commands in `docs/202606081640-pp-doclayoutv3-ggml-engine.md`; do not duplicate them here.

Last known passing checks:

- `cargo fmt`
- `cargo test`
- Python `py_compile` for both scripts
- `cmake --build target/native/pp-doclayout-ggml-build --target pp-doclayout-ggml -j4`
- Native `--summary --smoke`
- ONNXRuntime vs GGML diffs for first block and stem block

## Suggested Skills

- `karpathy-guidelines`: use while extending the executor, because incremental verified slices matter here.
- `handoff`: only when preparing the next continuation note.

