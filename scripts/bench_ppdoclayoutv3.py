#!/usr/bin/env python3
"""Benchmark PP-DocLayoutV3: native GGML engine (CLI) vs ONNXRuntime.

Pure external benchmark -- does not modify or instrument the inference code.

  * ONNXRuntime is driven in-process via onnxruntime (CPU / CoreML EPs).
  * The GGML engine is driven exactly as a user would: by invoking the
    `pp-doclayout-ggml` CLI as a subprocess. Each invocation loads the GGUF
    weights, parses the execution plan, builds the graph and runs one image,
    so its wall time is comparable to ONNXRuntime "cold" (new session + run).
    ONNXRuntime "warm" (reused session, run only) is reported for context --
    the CLI has no persistent-session mode.

Both backends receive the identical preprocessed image and the same
im_shape=[S,S] / scale_factor=[1,1] constants the engine hard-codes, so the
detection outputs (fetch_name_0, [300,7]) can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
def preprocess(image_path: Path, size: int) -> np.ndarray:
    """Resize SxS (bilinear), scale to [0,1], RGB, NCHW float32.

    Mirrors PaddleDetection Resize(keep_ratio=False) + NormalizeImage(
    is_scale=True, mean=0, std=1, norm_type=none) + Permute.
    """
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "runs": n,
        "min": s[0],
        "p50": statistics.median(s),
        "p90": s[min(n - 1, int(round(0.9 * (n - 1))))],
        "mean": statistics.fmean(s),
        "max": s[-1],
    }


# --------------------------------------------------------------------------- #
def bench_ort_warm(sess, out_names, feeds, runs, warmup):
    for _ in range(warmup):
        sess.run(out_names, feeds)
    samples, last = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = sess.run(out_names, feeds)
        samples.append((time.perf_counter() - t0) * 1e3)
    return stats(samples), last


def bench_ort_cold(onnx_path, ep, feeds, runs, warmup):
    import onnxruntime as ort
    samples = []
    for i in range(runs + warmup):
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(onnx_path), providers=[ep])
        out_names = [o.name for o in sess.get_outputs()]
        sess.run(out_names, feeds)
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            samples.append(dt)
    return stats(samples)


def bench_ggml_cli(binary, manifest, weights, plan, output_name,
                   input_f32, out_f32, runs, warmup):
    cmd = [str(binary), "--manifest", str(manifest), "--weights", str(weights),
           "--plan", str(plan), "--run-prefix", output_name,
           "--input-f32", str(input_f32), "--output-f32", str(out_f32)]
    samples = []
    for i in range(runs + warmup):
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = (time.perf_counter() - t0) * 1e3
        if res.returncode != 0:
            raise RuntimeError(f"GGML CLI failed:\n{res.stderr}\n{res.stdout}")
        if i >= warmup:
            samples.append(dt)
    return stats(samples), np.fromfile(out_f32, dtype=np.float32).reshape(-1, 7)


# --------------------------------------------------------------------------- #
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + \
        max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def parity(ref, cand, thr=0.5):
    """Match ref detections (score>thr) to nearest cand box by IoU."""
    r = ref[ref[:, 1] > thr]
    c = cand[cand[:, 1] > thr]
    matched, box_err, score_err, label_mismatch = 0, 0.0, 0.0, 0
    for row in r:
        if len(c) == 0:
            break
        ious = np.array([iou(row[2:6], cr[2:6]) for cr in c])
        j = int(ious.argmax())
        if ious[j] > 0.5:
            matched += 1
            box_err = max(box_err, float(np.abs(row[2:6] - c[j][2:6]).max()))
            score_err = max(score_err, abs(float(row[1] - c[j][1])))
            if int(round(row[0])) != int(round(c[j][0])):
                label_mismatch += 1
    return {"ref_dets": int(len(r)), "cand_dets": int(len(c)),
            "matched": matched, "max_box_px_err": box_err,
            "max_score_err": score_err, "label_mismatch": label_mismatch}


def fmt_table(rows):
    out = ["| Backend | Mode | runs | p50 (ms) | p90 (ms) | mean (ms) | min (ms) | it/s (p50) |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        thr = 1000.0 / r["p50"] if r["p50"] else 0.0
        out.append(f"| {r['backend']} | {r['mode']} | {r['runs']} | {r['p50']:.1f} | "
                   f"{r['p90']:.1f} | {r['mean']:.1f} | {r['min']:.1f} | {thr:.2f} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ggml-cpu", type=Path)
    ap.add_argument("--ggml-metal", type=Path)
    ap.add_argument("--output-name", default="fetch_name_0")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ggml-runs", type=int, default=10, help="GGML CLI reruns (slower, fewer)")
    ap.add_argument("--ort-eps", nargs="*",
                    default=["CPUExecutionProvider", "CoreMLExecutionProvider"])
    ap.add_argument("--workdir", type=Path, default=Path("artifacts/bench"))
    ap.add_argument("--md-out", type=Path, default=Path("artifacts/bench/report.md"))
    args = ap.parse_args()

    import onnxruntime as ort
    args.workdir.mkdir(parents=True, exist_ok=True)
    chw = preprocess(args.image, args.size)
    input_f32 = args.workdir / "input.f32"
    chw.tofile(input_f32)
    feeds = {"image": chw,
             "im_shape": np.array([[float(args.size)] * 2], np.float32),
             "scale_factor": np.array([[1.0, 1.0]], np.float32)}
    print(f"input {args.image.name} -> {chw.shape}  ({chw.nbytes/1e6:.1f} MB)")

    rows, refs = [], {}
    avail = ort.get_available_providers()

    for ep in args.ort_eps:
        if ep not in avail:
            print(f"[skip] {ep} unavailable")
            continue
        short = ep.replace("ExecutionProvider", "")
        sess = ort.InferenceSession(str(args.onnx), providers=[ep])
        out_names = [o.name for o in sess.get_outputs()]
        st_warm, last = bench_ort_warm(sess, out_names, feeds, args.runs, args.warmup)
        rows.append({"backend": f"ONNXRuntime {short}", "mode": "warm (run only)", **st_warm})
        refs[ep] = np.asarray(last[0], np.float32)
        st_cold = bench_ort_cold(args.onnx, ep, feeds, max(3, args.ggml_runs), args.warmup)
        rows.append({"backend": f"ONNXRuntime {short}", "mode": "cold (load+run)", **st_cold})
        print(f"[ort {short}] warm p50={st_warm['p50']:.1f}ms  cold p50={st_cold['p50']:.1f}ms")

    for label, binary in [("GGML CPU", args.ggml_cpu), ("GGML Metal", args.ggml_metal)]:
        if not binary:
            continue
        out_f32 = args.workdir / f"{label.replace(' ', '_')}_out.f32"
        st, det = bench_ggml_cli(binary, args.manifest, args.weights, args.plan,
                                 args.output_name, input_f32, out_f32,
                                 args.ggml_runs, max(1, args.warmup))
        rows.append({"backend": label, "mode": "CLI invocation (load+run)", **st})
        refs[label] = det
        print(f"[{label}] p50={st['p50']:.1f}ms")

    table = fmt_table(rows)
    par_lines = []
    if "CPUExecutionProvider" in refs:
        ref = refs["CPUExecutionProvider"]
        par_lines = ["", "### Parity vs ONNXRuntime CPU (IoU>0.5 matched detections)", "",
                     "| Backend | ref dets | cand dets | matched | max box err (px) | max score err | label mism. |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
        for label in ("GGML CPU", "GGML Metal", "CoreMLExecutionProvider"):
            cand = refs.get(label)
            if cand is None:
                continue
            p = parity(ref, cand)
            name = label.replace("ExecutionProvider", " (ORT CoreML)")
            par_lines.append(f"| {name} | {p['ref_dets']} | {p['cand_dets']} | {p['matched']} | "
                             f"{p['max_box_px_err']:.3f} | {p['max_score_err']:.2e} | {p['label_mismatch']} |")

    report = (f"## PP-DocLayoutV3 — GGML engine vs ONNXRuntime\n\n"
              f"- image: `{args.image.name}`  ·  input: {args.size}×{args.size}  "
              f"·  output: `{args.output_name}` [300,7]\n\n{table}\n" + "\n".join(par_lines) + "\n")
    print("\n" + report)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(report, encoding="utf-8")
    (args.workdir / "bench_rows.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
