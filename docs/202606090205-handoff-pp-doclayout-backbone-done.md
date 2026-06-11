# Handoff: PP-DocLayoutV3 GGML — CNN Backbone Complete

Continuation of `docs/202606090136-handoff-pp-doclayoutv3-ggml.md`.

## Goal (unchanged)

Local PP-DocLayoutV3 inference in GGML/C++ for the Tauri app, proven by staged
parity against ONNXRuntime. `--detect` still intentionally gated; goal not complete.

## What changed this session

- Built a generalized executor `--run-prefix` over `pp-doclayoutv3.plan.json`.
  Replaces the hand-written block runners; walks nodes in order and builds the
  GGML graph for a supported op subset up to a named output.
  - New code in `native/pp-doclayout-ggml/pp_doclayout_ggml.cpp`: a minimal JSON
    parser (`JsonValue`/`JsonParser`), plan-node structs, attr helpers,
    `same_upper_pad`, an elementwise `erf_custom_op`, and
    `Engine::run_plan_prefix(...)`.
  - Header + CLI wired: `--plan`, `--run-prefix <output_name>`, `--input-f32`.
- 20 supported ops: `Conv` (group=1 + depthwise via `ggml_conv_2d_dw_direct`),
  `BatchNormalization` (host-folded), `Relu`, `MaxPool`, `Concat` (float + int),
  `Identity`, `Add`/`Mul`/`Sub`/`Div`, `Sigmoid`, `Erf` (`ggml_map_custom1`),
  `Softmax`, `LayerNormalization`, `MatMul` (`mul_mat(cont(transpose(B)),A)`),
  `Shape`/`Slice`/`Reshape`/`Transpose` (build-time concrete shapes),
  `Resize` (nearest + bilinear via `ggml_interpolate`, scales or dynamic sizes).
- `Concat` is rank-aware (ONNX axis -> `(rank-1)-axis`), handling rank-3 sequence
  concats in the neck as well as rank-4 NCHW concats in the backbone.
- Decoder shape/elementwise ops added: `Squeeze`/`Unsqueeze` (reshape across rank),
  `Expand`, `Cast` (f32 no-op), `Where` (`Y+cond·(X−Y)`),
  `ReduceMax`/`ReduceMin`/`ReduceSum` (`ggml_custom_4d` over last axis); `Identity`
  now aliases integer shape vectors too.
- Decoder data-dependent ops added: `TopK` (`ggml_argsort_top_k`, indices threaded
  as a gather index), `GatherND` (`ggml_get_rows`, batch=1), `Range`/`Tile`,
  `Greater`/`Clip`/`Log`/`Split`, and `ReduceMax/Min/Sum` over any trailing axes.
- `GridSample` (bilinear/zeros/align_corners=0) implemented as a `ggml_custom_4d`
  op (`grid_sample_op`); blocked behind a 6D reshape (see Next work).
- Verified vs ONNX through: full CNN backbone (node 325), full first transformer
  encoder block incl. erf-GELU (node 378), the entire FPN/HybridEncoder neck
  `[1,13125,256]` (node 619), and into the RT-DETR decoder through the masked
  encoder-output projection (642), query selection `TopK`+`GatherND` (664, exact
  value+order), and the per-query bbox MLP (676). See gate table in engine notes.
- `--run-prefix p2o.pd_op.relu.3.0` reproduces hard-coded `--stem-block` bit-for-bit.

Authoritative detail (full op map + verified-gate table + next steps) is in
`docs/202606081640-pp-doclayoutv3-ggml-engine.md`.

## State of the tree

Committed on branch `pp-doclayout-ggml-backbone`. `test.sh` (personal launcher)
left untracked and uncommitted. `cargo fmt --check` clean; `cargo test -p
archon-core` 3/3; native build + smoke pass.

## Where the boundary is now

Verified watermark: node 676 (`p2o.pd_op.add.40.0`, the per-query bbox MLP).
Query selection (`TopK`+`GatherND`, node 664) matches ONNX in value AND order
(max 9.3e-6, 0/300 rows differing); the bbox MLP and the batched matmul
(`bmm.0.0`, rel err 5e-5) are correct.

**Resolved (not a bug)** — the deformable-attention valid-mask / reference-point
logic (nodes 694–798) shows a ~2e-2 discrepancy at `p2o.pd_op.divide.0.0` (798)
→ reference points `p2o.pd_op.sigmoid.19.0` (841, 122/300 rows). Root cause: node
697 `Greater` thresholds the attention map at `> 0`, and exactly **2 of 12M**
values are within |v|≈1.4e-6 of zero; ONNXRuntime vs GGML F32 matmul summation
order differs by more than that, flipping those 2, which shifts the mask→box
`ReduceMin` by a few of the 200×200 grid cells. Inherent FP threshold sensitivity,
not an op error (`ReduceMax.3` at 705 is bit-exact). Don't chase bit-exact gates
here — validate end-to-end at the final boxes/labels, where the 6 refinement
layers should wash it out.

## Next work (priority order)

1. **Deformable-attention >4D blocker (current edge).** `GridSample` (bilinear/
   zeros/align_corners=0) is implemented (`ggml_custom_4d`, `grid_sample_op`) but
   unreachable: node 910 `Reshape.51` targets 6D `[1,300,8,3,4,2]` (batch, query,
   head, level, point, xy) and GGML is 4D-max. Either special-case the MSDeformAttn
   block (batch=1 ⇒ collapse to the 4D form GridSample needs: data `[8,32,H,W]`,
   grid `[8,300,L·P,2]`, out `[8,32,300,L·P]`), or fuse the whole sample+aggregate
   into one custom op. `Reshape` now reports a clear size mismatch on >4D targets.
2. Remaining decoder layers ×6 (~node 665–2600): query self-attention (ops exist)
   + the deformable cross-attention above.
3. Postprocess (~node 2600+): `Einsum`, `ScatterND`, `Floor`, `Mod`, `Max`,
   final `TopK` → boxes/labels.
4. Switch `--run-prefix` allocation to static-ctx-weights + `ggml_gallocr`
   compute graph before full-graph runs (memory; currently allocates all tensors).
5. Only flip app ingestion after `--detect` matches ONNX labels/boxes in tolerance.

Test fixture: `testdata/brandt-et-al-2017-wto-accession.pdf` (real multi-column
paper) for end-to-end layout detection once `--detect` works.

Note: node 622 `Slice` has 5 inputs (data, starts, ends, axes, steps) on a shape
vector; the current shape-vector `Slice` reads 3 + optional axes (step assumed 1),
which is fine here but generalize if a non-unit step shows up.

## Verification quickstart

```bash
PY=/Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
# reference (writes matching --input-f32)
PYTHONPATH=/tmp/onnx-inspect-py312 "$PY" scripts/pp_doclayoutv3_prefix_reference.py \
  --onnx /Users/weixu/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --output-name <NAME> --input-out target/native/x-in.f32 \
  --reference-out target/native/x-ref.f32 --shape-out target/native/x-shape.txt
# native
target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights target/native/pp-doclayoutv3-f32.gguf \
  --plan target/native/pp-doclayoutv3.plan.json \
  --run-prefix <NAME> --input-f32 target/native/x-in.f32 \
  --output-f32 target/native/x-got.f32
# diff: numpy abs(ref-got) max/mean
```

Build (if needed):

```bash
cmake --build target/native/pp-doclayout-ggml-build --target pp-doclayout-ggml -j4
```

## Suggested skills

- `karpathy-guidelines`: keep extending one verified slice at a time.
- `handoff`: only when writing the next continuation note.
