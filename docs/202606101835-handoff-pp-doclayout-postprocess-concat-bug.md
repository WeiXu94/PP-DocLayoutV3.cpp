# Handoff: PP-DocLayoutV3 GGML — Postprocess Ops Implemented, One Concat Bug Remaining

Continuation of `docs/202606091634-handoff-pp-doclayout-decoder-msdeform.md`.

## What changed this session

### Full RT-DETR decoder executes (prior milestone, committed earlier)
- 6 MSDeformAttn blocks run as fused CPU ops (`msdeform_attn_op`) — the 4-D tensor
  wall was resolved by fusing the sample+weight+sum of each decoder layer.
- Verified: fused op exact to 3.6e-7; full layer-0 with clean refpts 1.8e-6; all
  6 layers run to `layer_norm.21.0` (node 2533). Drift beyond that is a known
  FP-threshold issue, not an op bug.

### Postprocess ops implemented (this session)
5 new ops added to `native/pp-doclayout-ggml/pp_doclayout_ggml.cpp`:

| Op | Implementation |
|---|---|
| Floor | `ggml_map_custom1` with `floorf` |
| Mod | float-domain: `x - floor(x/div)*div` using ggml ops |
| Einsum `bij,bjk->bik` | `ggml_mul_mat(A, B)` — both have contracted dim at ne[0] |
| ScatterND | `ggml_custom_4d`: scatter TopK scores into a class-histogram canvas; built-in `indices % divisor` logic |
| Max | int-vector elementwise `max(a[i], b[i])` |

Plus extensive infrastructure co-developed with these ops:
- **Float Slice**: axis 0..rank-1, 5-D collapsed-leading-dims handling
- **Float Tile**: `ggml_repeat_4d` with multiplies
- **Non-trailing ReduceSum**: permute+reduce+unpermute for single non-trailing axis
- **I32 ↔ F32 conversion** (`i32_to_f32_op`, `f32_to_i32_op`) — GGML has no type cast
- **TopK** now produces both values (`ggml_get_rows`) and indices
- **Cast** handles I32/I64 → F32; **GatherND** auto-converts F32→I32 for `ggml_get_rows`
- **Expand**: materializes int-vectors as F32 tensors
- **Concat**: materializes int/gather-index inputs as F32 with scalar replication;
  passthrough for pure int/gather concats restores correct int-vector merging
- **Graph inputs** `im_shape` / `scale_factor` registered as F32 `[800,800]` / `[1,1]`
- `--inject <name>=<file.f32>` CLI for isolating slices from upstream drift

### Verified gates
All individual gates between the decoder exit and the detection output execute correctly:
- `einsum.0.0` (node 2617) — the score matrix
- `multiply.34.0` (node 2686) — box decode to pixel coords
- `sigmoid.27.0`, `sigmoid.26.0` — class/instance scores
- `remainder.0.0` (Mod result, class ID per query)

## The one remaining bug

**Concat.220 (node 2774)** fails with `ggml_concat` ne[d] mismatch. This concat
builds the `[batch_idx, class_idx]` scatter-fetch indices for the final output.

Inputs:
1. `helper.unsqueeze.274` — scalar int-vector (batch index = 0, 1 element). Must
   be replicated to 300 elements to match the other input.
2. `helper.unsqueeze.275` — I32 tensor from TopK (class indices, 300 elements).

Both are classified as `!any_float` (no clean float tensor among inputs), so the
new passthrough path fires for query-selection-index-building concats. But this
particular concat truly NEEDS F32 materialization because it feeds ScatterND
and GatherND which ultimately drive `ggml_get_rows`. The scalar→300 replication
code exists (in the float materialization path) and computes the right
`target_size`, but the resulting ggml tensor's shape doesn't propagate correctly
through the concat because the first_shape is derived from the I32 index tensor's
4-D layout rather than the intended 3-D output layout.

**Root cause:** The `any_float` gate misclassifies this concat as "pure int/
gather" when it should use the full materialization path. The fix is to let this
concat go through the float path with correct rank tracking.

**Fix approach:** Change the `any_float` check to also trigger float
materialization when at least one input is a `gather_index` entry (not just
clean float). Or, more robustly, always use the float path when the concat axis
and rank from `attr_int(attrs, "axis")` suggest the output must be a multi-dim
tensor consumed by downstream float ops (like GatherND).

Once this is fixed, the remaining path through `fetch_name_0` uses only already-
implemented ops (Concat → Unsqueeze → GatherND → Sigmoid → output assembly).

## Key files

| File | Role |
|---|---|
| `native/pp-doclayout-ggml/pp_doclayout_ggml.cpp` | ~3000 lines, the entire executor |
| `native/pp-doclayout-ggml/main.cpp` | CLI, argument parsing, `--inject` |
| `native/pp-doclayout-ggml/pp_doclayout_ggml.hpp` | Header, `Engine` class, `PrefixRunResult` |
| `docs/202606081640-pp-doclayoutv3-ggml-engine.md` | Authoritative engine notes (op table, gate table) |
| `docs/202606091634-handoff-pp-doclayout-decoder-msdeform.md` | Previous handoff (deformable attention) |
| `target/native/pp-doclayoutv3.plan.json` | The 2818-node execution plan |
| `target/native/pp-doclayoutv3-f32.gguf` | 128 MB weight file |
| `scripts/pp_doclayoutv3_prefix_reference.py` | ONNXRuntime reference generator |
| `testdata/brandt-et-al-2017-wto-accession.pdf` | End-to-end test fixture |

## Build & verify

```bash
cmake -S native/pp-doclayout-ggml -B target/native/pp-doclayout-ggml-build \
  -DGGML_DIR=/tmp/badlogic-ggml-inspect/ggml -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build target/native/pp-doclayout-ggml-build --target pp-doclayout-ggml -j4

# Generate reference inputs
PYTHONPATH=/tmp/onnx-inspect-py312 \
  target/native/pp-doclayout-ggml-build/pp-doclayout-ggml ... see engine notes

# Test a gate
target/native/pp-doclayout-ggml-build/pp-doclayout-ggml \
  --manifest target/native/pp-doclayoutv3.manifest.json \
  --weights  target/native/pp-doclayoutv3-f32.gguf \
  --plan     target/native/pp-doclayoutv3.plan.json \
  --run-prefix <tensor_name> --input-f32 <input>.f32 --output-f32 <out>.f32

# Inject clean upstream values to isolate a slice
  --inject <name>=<file.f32>
```

The GGML checkout at `/tmp/badlogic-ggml-inspect` is `github.com/ggml-org/llama.cpp`
(cloned fresh this session; `ggml/` subdirectory used as `GGML_DIR`).

## Next steps (priority)

1. **Fix Concat.220 dimension mismatch.** See "The one remaining bug" above.
2. **Verify `fetch_name_0` end-to-end.** Once the concat passes, the output
   `[N, 7]` (detection boxes + labels) should compute. Compare against ONNX
   using `--run-prefix fetch_name_0`.
3. **Decide end-to-end parity tolerance.** The known FP-threshold issue means
   intermediate gates won't be bit-exact. Validate at the final detection
   output (IoU / label match), not raw tensors.
4. *(Later)* Metal kernels for custom ops. *(Later)* Memory optimization
   (static ctx + ggml_gallocr). *(Final)* Flip app ingestion.

## Suggested skills
- `karpathy-guidelines`: make surgical changes, verify one gate at a time.
- No other skills needed.
