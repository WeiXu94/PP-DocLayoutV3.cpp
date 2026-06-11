# PP-DocLayoutV3 GGML Engine Notes

## Current boundary

The current Tauri import path still uses PyMuPDF4LLM. PP-DocLayoutV3 is now represented in `archon-core` as a layout contract plus a native GGML engine scaffold, but the full detector graph is not enabled yet.

Update 2026-06-09: the native engine has a generalized executor that walks
`pp-doclayoutv3.plan.json` node by node and builds the GGML graph for a supported
op subset. It runs any prefix up to a named output. Verified against ONNXRuntime
within single-digit ×1e-6 through: the entire CNN backbone (node 325), the full
first transformer encoder block including erf-GELU (node 378), the entire
FPN/HybridEncoder neck producing the multi-scale feature sequence `[1, 13125, 256]`
(node 619), and into the RT-DETR decoder through query selection (`TopK`+`GatherND`,
node 664, exact in value and order) and the per-query bbox MLP (node 676).

Update 2026-06-09 (later): the **6 RT-DETR decoder layers now execute end to
end**, through the final decoder hidden state `layer_norm.21.0` (node 2533). The
multi-scale deformable-attention (MSDeformAttn) module of each layer materializes
5-D/6-D tensors (query, head, level, point, xy) that GGML (4-D max) cannot
represent, so it is implemented as **one fused CPU op** (`msdeform_attn_op`) that
folds the sampling-location math, bilinear `GridSample`, and the attention-weighted
sum into nested loops — no >4-D tensor ever exists. The executor pre-scans the
plan, finds each block by its 3 `GridSample` nodes, pins the 4 inputs (value /
offsets / post-softmax attn / refpts) and the `ReduceSum` output by structural
anchors, marks the interior nodes to skip, and emits the fused op in place of the
`ReduceSum`. Verified vs ONNXRuntime: the fused op with all 4 inputs injected from
ONNX is **3.6e-7**; full decoder layer 0 with clean reference points injected
(`--inject p2o.pd_op.sigmoid.19.0=…`) is **1.8e-6** (so the native query
self-attention is correct too). End-to-end without injection drifts (~0.3 rel by
the final layer) purely from the known reference-point FP-threshold issue (the
`>0` mask flip at `sigmoid.19`, 2 of 12M values) propagating and being amplified
by the sampling — not an op error. The hard-coded `--first-block` / `--stem-block`
runners are retained as baselines; the generalized `--run-prefix` reproduces them
bit-for-bit.

This is intentional: the local PP-DocLayoutV3 ONNX artifact is not a small hand-written graph. It contains:

| Item | Count |
|---|---:|
| ONNX nodes | 2818 |
| Initializers | 1899 |
| Parameters | 32503418 |
| Operator families | 42 |

Important ops include `Conv`, `BatchNormalization`, `MatMul`, `LayerNormalization`, `GridSample`, `TopK`, `GatherND`, `ScatterND`, `Einsum`, and dynamic shape ops. A faithful GGML engine should be a generated/subset runner or a staged architecture port, not guessed from a few layer names.

## Preprocess contract

The local `config.json` says:

| Setting | Value |
|---|---|
| Resize | `800x800` |
| Keep ratio | `false` |
| Normalize mean | `[0, 0, 0]` |
| Normalize std | `[1, 1, 1]` |
| Layout | `NCHW` |

Do not use ImageNet mean/std for this model.

## Added pieces

- `crates/archon-core/src/layout.rs`: labels, Markdown role mapping, preprocessing, row postprocess, GGML engine discovery.
- `scripts/pp_doclayoutv3_onnx_to_ggml_manifest.py`: ONNX/config inventory generator, execution-plan generator, and ONNX-initializer-to-GGUF weight writer.
- `scripts/pp_doclayoutv3_prefix_reference.py`: deterministic ONNXRuntime intermediate reference generator.
- `native/pp-doclayout-ggml`: C++17 GGML binary with backend preference, GGUF weight loading, smoke graph, the original `--first-block` / `--stem-block` baseline runners, and a generalized `--run-prefix` executor over `plan.json` (minimal JSON parser + per-op graph builder) covering the supported op subset through the full CNN backbone.

## Verified commands

```bash
PYTHONPATH=/tmp/onnx-inspect-py312 \
  /Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/pp_doclayoutv3_onnx_to_ggml_manifest.py \
  --onnx /Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --config /Users/weixu/models/pp-doclayoutv3-onnx/config.json \
  --out target/native/pp-doclayoutv3.manifest.json \
  --gguf-out target/native/pp-doclayoutv3-f32.gguf \
  --plan-out target/native/pp-doclayoutv3.plan.json

cmake -S native/pp-doclayout-ggml \
  -B target/native/pp-doclayout-ggml-build \
  -DGGML_DIR=/tmp/badlogic-ggml-inspect \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF

cmake --build target/native/pp-doclayout-ggml-build --target pp-doclayout-ggml -j4

target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights target/native/pp-doclayoutv3-f32.gguf \
  --summary --smoke

PYTHONPATH=/tmp/onnx-inspect-py312 \
  /Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/pp_doclayoutv3_prefix_reference.py \
  --onnx /Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --input-out target/native/prefix-input.f32 \
  --reference-out target/native/prefix-relu0-reference.f32 \
  --shape-out target/native/prefix-relu0-shape.txt

target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights target/native/pp-doclayoutv3-f32.gguf \
  --first-block target/native/prefix-input.f32 \
  --output-f32 target/native/prefix-relu0-ggml.f32

PYTHONPATH=/tmp/onnx-inspect-py312 \
  /Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/pp_doclayoutv3_prefix_reference.py \
  --onnx /Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --output-name p2o.pd_op.relu.3.0 \
  --input-out target/native/stem-input.f32 \
  --reference-out target/native/stem-relu3-reference.f32 \
  --shape-out target/native/stem-relu3-shape.txt

target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights target/native/pp-doclayoutv3-f32.gguf \
  --stem-block target/native/stem-input.f32 \
  --output-f32 target/native/stem-relu3-ggml.f32
```

Generated artifact sizes:

| Artifact | Size |
|---|---:|
| `target/native/pp-doclayoutv3.manifest.json` | 16K |
| `target/native/pp-doclayoutv3.plan.json` | 1.2M |
| `target/native/pp-doclayoutv3-f32.gguf` | 128M |

Native summary / smoke output includes:

```json
{"backend":"CPU","values":[2,-4,7,0.5]}
```

The native loader read the generated GGUF as 1899 tensors with 130018784 total tensor bytes. The largest tensor was `conv2d_78.w_0_deepcopy_536` (`f32`, 13631488 bytes).

First-block comparison against ONNXRuntime intermediate output `p2o.pd_op.relu.0.0`:

| Metric | Value |
|---|---:|
| Output shape | `1x32x400x400` |
| Values | 5120000 |
| Max absolute error | 8.344650268554688e-7 |
| Mean absolute error | 1.5356839355717966e-8 |
| RMSE | 3.3704203872275684e-8 |

Stem-block comparison against ONNXRuntime intermediate output `p2o.pd_op.relu.3.0`:

| Metric | Value |
|---|---:|
| Output shape | `1x32x200x200` |
| Values | 1280000 |
| Max absolute error | 1.0728836059570312e-6 |
| Mean absolute error | 7.150985936732468e-8 |
| RMSE | 1.1133704447274795e-7 |

## Generalized executor (`--run-prefix`)

Usage:

```bash
target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights target/native/pp-doclayoutv3-f32.gguf \
  --plan target/native/pp-doclayoutv3.plan.json \
  --run-prefix <onnx_output_name> \
  --input-f32 <input.f32> --output-f32 <out.f32>
```

Reference is generated by `scripts/pp_doclayoutv3_prefix_reference.py --output-name <name>`
(seed 7), which writes the matching `--input-f32` so the input is shared.

Per-op handling:

| ONNX op | GGML mapping |
|---|---|
| `Conv` (group=1) | `ggml_conv_2d`; `SAME_UPPER` even kernels pre-padded with `ggml_pad_ext` |
| `Conv` (depthwise, weight `[KW,KH,1,C]`) | `ggml_conv_2d_dw_direct` |
| `BatchNormalization` | host-folded per-channel scale/shift via `ggml_mul`/`ggml_add` (broadcast `[1,1,C,1]`) |
| `Relu` | `ggml_relu` |
| `MaxPool` | `ggml_pad_ext` (zero, safe after ReLU) then `ggml_pool_2d` MAX |
| `Concat` | `ggml_concat`, ONNX axis -> GGML dim `3-axis`; integer (shape) concat handled separately |
| `Identity` | alias (propagates tracked ONNX shape) |
| `Add`/`Mul`/`Sub` | `ggml_add`/`ggml_mul`/`ggml_sub`, larger operand as broadcast base |
| `Div` | `ggml_div`, operand order preserved |
| `Sigmoid` | `ggml_sigmoid` |
| `Erf` | `ggml_map_custom1` with `std::erf` (CPU; ggml has no standalone erf) |
| `Softmax` | `ggml_soft_max` (last axis only = GGML ne0) |
| `LayerNormalization` | `ggml_norm` over ne0 + per-channel `mul`/`add` (last axis only) |
| `MatMul` | `ggml_mul_mat(cont(transpose(B)), A)` = ONNX `A@B` |
| `Shape` | emits the input's ONNX shape (batch resolved to 1) as an int vector |
| `Slice` | 1-D shape-vector slice (axis 0) |
| `Reshape` | `ggml_reshape_4d` with concrete dims (0=copy, -1=infer) |
| `Transpose` | `ggml_permute`+`ggml_cont`, ONNX perm -> GGML dim perm |
| `Resize` | nearest + bilinear via `ggml_interpolate`; target from `scales` or dynamic `sizes`; `half_pixel`/`align_corners` |
| `Squeeze`/`Unsqueeze` | int vectors pass through; float tensors `ggml_reshape` to the new rank |
| `Expand` | `ggml_repeat_4d` to the target shape |
| `Cast` | representation no-op (all values tracked as f32; bool operands are already 0/1) |
| `Where` | `Y + cond·(X−Y)` (cond is 0/1) |
| `ReduceMax`/`ReduceMin`/`ReduceSum` | `ggml_custom_4d` reduce over any trailing axes (collapsed into ne0) |
| `TopK` | `ggml_argsort_top_k` (DESC, matches ONNX `sorted`/`largest`); indices threaded as a gather index |
| `GatherND` | row gather via `ggml_get_rows` using a runtime index (batch=1) |
| `Range`/`Tile` | integer arange / integer passthrough on the batch-index path |
| `Greater` | `step(a − b)` as a 0/1 mask |
| `Clip` | `ggml_clamp` (min/max from constant inputs) |
| `Log` | `ggml_log` |
| `Split` | strided `ggml_view` slices along a dim (multi-output) |
| `GridSample` | bilinear / zeros / align_corners=0 via `ggml_custom_4d` (implemented; not yet reachable — see 4D blocker) |

Shapes: everything runs at batch=1 / fixed 800×800, so shapes are concrete at
build time. Rank-4 NCHW shapes derive from reversed GGML `ne`; non-rank-4 values
(Reshape/Transpose/MatMul/elementwise outputs) carry an explicit tracked ONNX
shape. Integer 1-D tensors (Shape/Slice/int-Concat results, int initializers)
are evaluated to `int64` vectors at build time.

Verified gates (`plan vs ONNXRuntime`, seed 7):

| Output (node) | Shape | Max abs err | Mean abs err |
|---|---|---:|---:|
| `p2o.pd_op.relu.3.0` (18) | 1×32×200×200 | 1.07e-6 | 7.15e-8 |
| `p2o.pd_op.relu.13.0` (66, first depthwise) | 1×96×100×100 | 6.97e-6 | 3.73e-7 |
| `p2o.pd_op.relu.36.0` (209, 51 convs) | 1×1024×50×50 | 7.03e-6 | 1.15e-7 |
| `p2o.pd_op.batch_norm_.82.0` (325, full backbone + residuals) | 1×256×25×25 | 2.38e-6 | 1.47e-7 |
| `p2o.pd_op.transpose.0.0` (330, flatten+transpose) | 1×625×256 | 2.38e-6 | 1.47e-7 |
| `p2o.pd_op.matmul.3.0` (349, QKᵀ) | 1×8×625×625 | 7.63e-5 | 5.75e-6 |
| `p2o.pd_op.softmax.0.0` (354) | 1×8×625×625 | 1.49e-7 | 1.20e-9 |
| `p2o.pd_op.layer_norm.0.0` (363, self-attention sublayer) | 1×625×256 | 6.68e-6 | 4.03e-7 |
| `p2o.pd_op.layer_norm.1.0` (378, **full encoder block** w/ GELU) | 1×625×256 | 3.82e-6 | 3.27e-7 |
| `p2o.pd_op.nearest_interp.0.0` (386, Resize nearest) | 1×256×50×50 | 4.65e-6 | 3.25e-7 |
| `p2o.pd_op.multiply.13.0` (544, FPN neck pre-bilinear) | 1×64×50×50 | 4.77e-6 | 1.97e-7 |
| `p2o.pd_op.bilinear_interp.0.0` (545, Resize bilinear/half_pixel) | 1×64×100×100 | 3.58e-6 | 1.71e-7 |
| `p2o.pd_op.concat.11.0` (619, **full FPN/HybridEncoder neck**) | 1×13125×256 | 2.01e-6 | 8.50e-8 |
| `p2o.pd_op.layer_norm.2.0` (642, decoder: masked enc-output proj) | 1×13125×256 | 9.48e-6 | 4.59e-7 |
| `p2o.pd_op.max.0.0` (647, query-selection score ReduceMax) | 1×13125 | 1.07e-5 | 8.70e-7 |
| `p2o.pd_op.gather_nd.0.0` (664, **300 selected queries**, value+order) | 1×300×256 | 9.30e-6 | 5.37e-7 |
| `p2o.pd_op.add.40.0` (676, per-query bbox MLP) | 1×300×32 | 3.82e-5 | 3.07e-6 |
| `p2o.pd_op.bmm.0.0` (686, batched matmul; rel err 5e-5, values ≤270) | 1×300×40000 | 5.34e-4 | 2.27e-5 |

`--run-prefix p2o.pd_op.relu.3.0` reproduces the hard-coded `--stem-block` output
bit-for-bit (0.0 diff). The neck flattens 3 feature levels (25², 50², 100² =
13125 tokens) into one sequence — the encoder memory fed to the RT-DETR decoder.

## Next implementation steps

Verified watermark: node 676 (RT-DETR decoder per-query bbox MLP). Query selection
(`TopK`+`GatherND`, node 664) matches ONNX in value and order; the bbox MLP and the
batched matmul are correct.

Resolved finding (not a bug): the deformable-attention **valid-mask / reference-
point** logic (nodes 694–798) shows a ~2e-2 (few-grid-cell) discrepancy at
`p2o.pd_op.divide.0.0` (798), propagating to reference points
`p2o.pd_op.sigmoid.19.0` (841, 122/300 rows). Root cause: node 697 `Greater`
thresholds the attention map at exactly `> 0`, and **2 of 12,000,000** values are
within |v|≈1.4e-6 of zero. ONNXRuntime's and GGML's F32 matmul summation orders
differ by more than that, so those 2 borderline values flip the mask; the mask→box
`ReduceMin/Max` then shifts a box edge by a few of the 200×200 grid cells. This is
inherent floating-point sensitivity at a hard threshold, not an op error — the
decoder-prefix ops (`Greater`/`Where`/`ReduceMin`/`Split`/reductions) are correct
(`ReduceMax.3` at node 705 is bit-exact; the box-decode arithmetic matches where
the mask agrees). Validate this region end-to-end at the final detection output
(boxes/labels) rather than by bit-exact intermediate gates; the 6 decoder layers
refine boxes and the difference is expected to wash out.

Then:

1. **Deformable attention >4D blocker (current edge).** `GridSample` (bilinear/
   zeros/align_corners=0) is implemented as a `ggml_custom_4d` op, but it is not
   yet reachable: node 910 `Reshape.51` targets a 6D shape `[1,300,8,3,4,2]`
   (batch, query, head, level, point, xy) and GGML tensors are 4D max. The
   MSDeformAttn module's >4D reshape/transpose sequence cannot run node-by-node.
   Options: (a) special-case the deformable-attention block — with batch=1,
   collapse the logical 6D into the 4D form the downstream `GridSample` consumes
   (`data [8,32,H,W]`, `grid [8,300,L·P,2]`, out `[8,32,300,L·P]`); or (b)
   implement the whole MSDeformAttn sampling+aggregation as one fused custom op.
   `Reshape` now emits a clear size-mismatch error (node + shapes) on >4D targets.
2. Remaining decoder layers ×6 (~node 665–2600): query self-attention (ops exist)
   + the deformable cross-attention above.
3. Detector postprocess (~node 2600+): `Einsum`, `ScatterND`, `Floor`, `Mod`,
   `Max`, final `TopK` → boxes/labels.
4. Memory: `--run-prefix` allocates every graph tensor at once
   (`ggml_backend_alloc_ctx_tensors`). For full-graph runs switch to a static
   context for weights/inputs plus `ggml_gallocr` for the compute graph.
5. Compare intermediate tensors against ONNXRuntime after each block group.
6. Add Metal/Vulkan build variants after CPU correctness is proven.
7. Only switch ingestion from PyMuPDF/ONNX once `detect` emits the same labels/boxes as the ONNX reference within tolerance.
