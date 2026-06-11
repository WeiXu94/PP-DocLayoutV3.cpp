# Handoff: PP-DocLayoutV3 GGML — Full Decoder Executes (MSDeformAttn fused)

Continuation of `docs/202606090205-handoff-pp-doclayout-backbone-done.md`.

## Goal (unchanged)

Local PP-DocLayoutV3 inference in GGML/C++ for the Tauri app, proven by staged
parity against ONNXRuntime. `--detect` still intentionally gated; not complete.

## What changed this session

- **Resolved the deformable-attention 4-D blocker.** RT-DETR MSDeformAttn needs
  5-D/6-D tensors (query, head, level, point, xy); GGML is 4-D max. Implemented
  the whole sample+weight+sum of each decoder layer as **one fused CPU op**,
  `msdeform_attn_op` (`ggml_custom_4d`), so no >4-D tensor exists:
  - inputs (all ≤4-D, executor-computed): value `reshape.23.0` `[1,S,8,32]`,
    offsets `add.63.0` `[1,300,192]`, post-softmax attn `softmax.2.0`
    `[1,300,8,12]`, refpts `unsqueeze.2.0` `[1,300,1,4]` (cx,cy,w,h).
  - math (graph-verified constants): `loc = ref_center + offset/4·ref_wh·0.5`;
    `grid = 2·loc−1`; align_corners=0 zeros-pad bilinear; spatial maps
    `[100²,50²,25²]` (split sizes `[10000,2500,625]`); output `[8,32,300]`.
- **Executor interception.** A pre-scan finds each of the 6 blocks by its 3
  `GridSample` nodes, pins the 4 inputs + `ReduceSum` output by structural
  anchors (Split→value, last Softmax→attn, 6-input-Concat-shaped Reshape→offsets,
  first 5-input Slice→refpts, first ReduceSum after the GridSamples→output), and
  marks interior nodes to skip. Skip = all-edge backward reachability from the
  ReduceSum, terminating at `Shape` (the float→int boundary, else the walk leaks
  upstream to the graph input) and at the 4 inputs; then a **keep pass** un-skips
  any interior dim-derivation helper still consumed by a kept node (e.g. the
  num-keys `Shape`/`Slice` that also feeds the value reshape). The fused op is
  emitted in place of the `ReduceSum` once all 4 inputs exist.
- **New CLI:** `--inject <value_name>=<file.f32>` overrides an activation with
  external f32 (same shape) to isolate a slice from upstream drift. Reusable for
  later layers / postprocess bring-up.

## Verification (seed 7, 800×800, vs ONNXRuntime)

| Gate | Result |
|---|---|
| Fused op, all 4 inputs injected from ONNX | **3.6e-7** (op exact) |
| Full decoder layer 0, clean `sigmoid.19` injected | **1.8e-6** (self-attn + op correct) |
| bbox MLP `add.40.0` (node 676), no-regression | 3.8e-5 |
| All 6 fused blocks execute → `layer_norm.21.0` (node 2533) | runs; max 3.64 |

The end-to-end drift (final layer ~0.3 rel, ~126/300 query columns) is **not an
op bug**: it is the single known reference-point FP-threshold issue — node 697
`Greater` thresholds the attention map at `>0`, 2 of 12M values are within
~1.4e-6 of zero, GGML vs ORT F32 matmul summation order flips them, shifting a
mask→box `ReduceMin` — propagating through `sigmoid.19` (the layer's reference
points, also the self-attn position embedding via `MatMul.14`) and amplified by
the deformable sampling. Confirmed by injecting clean `sigmoid.19.0`: layer-0
output collapses to 1.8e-6. Validate end-to-end at the final boxes, not here.

## Where the boundary is now

Verified watermark: full decoder layer 0 (1.8e-6 with clean refpts). All 6 layers
execute to `layer_norm.21.0` (node 2533). Drift downstream is the propagated
known FP issue, compounding per layer because each layer refines its own
reference points from the same `sigmoid` path.

## Next work (priority order)

1. **Postprocess (~node 2559→2818).** Remaining op families to reach final
   boxes/labels: `Einsum`, `ScatterND`, `Floor`, `Mod`, `Max`, and the final
   `TopK`. Wire them in `run_plan_prefix`, verify each slice with `--run-prefix`.
   (The final scores/boxes are where end-to-end parity actually matters; the
   refpt drift should wash out or be tolerable there.)
2. **Decide the final-parity tolerance** for `--detect`: compare native vs ONNX
   *labels + boxes* (post-NMS/topk), not intermediate tensors. Likely IoU/label
   match rather than raw-tensor 1e-5.
3. Switch `--run-prefix` allocation to static-ctx-weights + `ggml_gallocr` before
   full-graph runs (memory; currently allocates all tensors at once).
4. Only flip app ingestion (`testdata/brandt-et-al-2017-wto-accession.pdf` is the
   end-to-end fixture) after `--detect` matches ONNX labels/boxes in tolerance.

## Verification quickstart

```bash
PY=/Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
# /tmp/msdeform_dump.py writes /tmp/md/{image,value,offsets,attn,refpts,sum_ref}.f32
PYTHONPATH=/tmp/onnx-inspect-py312 "$PY" /tmp/msdeform_dump.py
BIN=target/native/pp-doclayout-ggml-build/pp-doclayout-ggml
M=(--manifest target/native/pp-doclayoutv3.manifest.json \
   --weights  target/native/pp-doclayoutv3-f32.gguf \
   --plan     target/native/pp-doclayoutv3.plan.json \
   --input-f32 /tmp/md/image.f32)
# fused-op gate (inject the 4 ONNX inputs)
"$BIN" "${M[@]}" --run-prefix p2o.pd_op.sum.0.0 --output-f32 /tmp/md/g.f32 \
  --inject p2o.pd_op.reshape.23.0=/tmp/md/value.f32 \
  --inject p2o.pd_op.add.63.0=/tmp/md/offsets.f32 \
  --inject p2o.pd_op.softmax.2.0=/tmp/md/attn.f32 \
  --inject p2o.pd_op.unsqueeze.2.0=/tmp/md/refpts.f32
```

Build: `cmake --build target/native/pp-doclayout-ggml-build --target pp-doclayout-ggml -j4`

## Suggested skills

- `karpathy-guidelines`: keep extending one verified slice at a time.
- `handoff`: when writing the next continuation note.
