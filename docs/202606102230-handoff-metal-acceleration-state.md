# Handoff — PP-DocLayoutV3 GGML Metal acceleration

Date: 2026-06-10. Continues the GGML-vs-ONNX benchmark work
([summary](202606102118-summary-ppdoclayoutv3-ggml-vs-onnx.md),
[plan](202606102140-plan-metal-custom-ops-patch.md)).

## Where we are

Benchmark + profiling done (see summary doc). Key facts driving the plan:
- GGML-CPU compute = ~1.9–2.2 s; **94% is ggml compute**, ~130 ms is build/upload.
- Warm engine **dropped** — would save only ~130 ms (decision: not worth it).
- Compute split: ~989 ms backbone (im2col+matmul convs — GPU-friendly) +
  ~1208 ms decoder (matmul/softmax/norm GPU-friendly; custom ops are not).
- GGML-CPU is ~6.5× slower than ORT-CPU (336 ms); the lever is GPU offload.

## Changes in the working tree (main repo only; ggml submodule clean)

`pp_doclayout_ggml.{hpp,cpp}`:
1. **RAM weight cache** — whole GGUF cached at load; `read_tensor_bytes` memcpy's
   from it. (Harmless; 0 ms warm benefit since OS page cache already served it.)
2. **`PPDOC_TIMING=1`** env → prints `graph_compute_ms` (gated, zero overhead off).
3. **`ggml_backend_sched` over [Metal, CPU]** in `run_plan_prefix_impl` (replaces
   single-backend `alloc_ctx_tensors` + `graph_compute`). Leaves flagged
   `ggml_set_input`; uploads skip tensors not in the requested subgraph.
   - Status: **runs end-to-end, numerically correct** on both build-cpu and
     build-metal (30/30 dets matched, max box err 4.72 px = same as before).
   - BUT **Metal is not used**: scheduler assigns all 3919 ops to CPU.

ggml fork work (reverted, to re-apply in the fork): `conv_2d_dw` Metal kernel
across 6 ggml-metal files — **validated** (depthwise output matched CPU ~1e-4).

## The blocker for real Metal speedup

ggml's scheduler offloads a matmul to the GPU only when its weight source lives in
a buffer tagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` and is host-resident
(`ggml-backend.cpp:916-928`). This engine allocates weights **inline in the compute
context**, so during graph-split the weights have no such buffer and the offload
check never fires → every op co-locates on CPU.

## Next step — weights-buffer refactor (prerequisite for any Metal speedup)

1. Create weight leaf tensors in a **separate ggml_context**; allocate them with
   `ggml_backend_alloc_ctx_tensors_from_buft(host_buft)` and
   `ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)`.
2. Keep only runtime inputs (`image`, `im_shape`, `scale_factor`) as
   `ggml_set_input`; do **not** flag weights as input.
3. Intermediates stay sched-galloc-allocated. With weights in a USAGE_WEIGHTS host
   buffer + `op_offload=true`, sched should offload backbone + decoder matmuls to
   Metal; custom ops fall back to CPU.
4. Then re-apply the `conv_2d_dw` Metal kernel in the ggml fork, then native Metal
   kernels for the custom ops (msdeform first).

## Expected payoff (set expectations)

Even with offload working, custom decoder ops stay on CPU (sched copies around
them). Realistic first-cut: offload the ~989 ms backbone + part of the decoder
matmuls → maybe ~1.5–2× over GGML-CPU, still slower than ORT-CPU until the custom
ops get native Metal kernels. ORT-CoreML was only ~1.3× over ORT-CPU on this model,
so the absolute Metal ceiling here is modest.

## Build & repro

```
cmake -S . -B build-metal -DGGML_DIR=ggml -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-metal -j8
PPDOC_TIMING=1 ./build-metal/pp-doclayout-ggml --manifest artifacts/v3_manifest.json \
  --weights artifacts/v3_weights.gguf --plan artifacts/v3_plan.json \
  --run-prefix fetch_name_0 --input-f32 artifacts/bench/input.f32 --output-f32 /tmp/o.f32
GGML_SCHED_DEBUG=2 ... 2>sched.txt   # inspect backend assignment
python3 scripts/bench_ppdoclayoutv3.py --onnx ~/models/pp-doclayoutv3-onnx/PP-DocLayoutV3.onnx \
  --image assets/doc_2col-01.png --plan artifacts/v3_plan.json --weights artifacts/v3_weights.gguf \
  --manifest artifacts/v3_manifest.json --ggml-cpu build-cpu/pp-doclayout-ggml
```
