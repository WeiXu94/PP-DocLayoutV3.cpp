#!/usr/bin/env python3
"""Run PP-DocLayoutV3 GGML warm-session layout detection over a PDF."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw


def bilinear_resample():
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), bilinear_resample())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def render_pdf(pdf_path: Path, dpi: int) -> list[Image.Image]:
    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    return images


def stats(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "min": s[0],
        "p50": statistics.median(s),
        "p90": s[min(n - 1, int(round(0.9 * (n - 1))))],
        "mean": statistics.fmean(s),
        "max": s[-1],
    }


def detections_from_output(rows: np.ndarray, labels: list[str], threshold: float) -> list[dict]:
    out = []
    for row in rows:
        score = float(row[1])
        if score < threshold:
            continue
        label_id = int(round(float(row[0])))
        label = labels[label_id] if 0 <= label_id < len(labels) else str(label_id)
        x1, y1, x2, y2 = [float(v) for v in row[2:6]]
        out.append({
            "label_id": label_id,
            "label": label,
            "score": score,
            "bbox_800": [x1, y1, x2, y2],
            "read_order": float(row[6]),
        })
    out.sort(key=lambda d: (d["read_order"], -d["score"]))
    return out


def annotate(img: Image.Image, detections: list[dict], size: int) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    sx = out.width / float(size)
    sy = out.height / float(size)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_800"]
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        draw.rectangle(box, outline=(255, 0, 0), width=3)
        text = f"{det['label']} {det['score']:.2f}"
        tx, ty = box[0], max(0, box[1] - 14)
        draw.rectangle([tx, ty, tx + max(80, 7 * len(text)), ty + 14], fill=(255, 255, 255))
        draw.text((tx + 2, ty), text, fill=(255, 0, 0))
    return out


def markdown_report(pdf: Path, cli: dict, pages: list[dict], labels: Counter, total_wall_ms: float) -> str:
    run_ms = [float(r["run_ms"]) for r in cli["runs"]]
    st = stats(run_ms)
    throughput = 1000.0 / st["p50"] if st["p50"] else 0.0
    lines = [
        "# PP-DocLayoutV3 Warm Metal PDF Batch",
        "",
        f"- pdf: `{pdf}`",
        f"- backend: `{cli.get('backend', '')}`",
        f"- pages: {len(pages)}",
        f"- prepare: {float(cli['prepare_ms']):.1f} ms",
        f"- wall total: {total_wall_ms:.1f} ms",
        "",
        "| Mode | pages | p50 run (ms) | p90 run (ms) | mean run (ms) | min (ms) | max (ms) | pages/s p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| GGML Metal warm batch | {len(pages)} | {st['p50']:.1f} | {st['p90']:.1f} | {st['mean']:.1f} | {st['min']:.1f} | {st['max']:.1f} | {throughput:.2f} |",
        "",
        "| Page | detections >= threshold | run (ms) | top labels |",
        "|---:|---:|---:|---|",
    ]
    for page, run in zip(pages, cli["runs"]):
        top = Counter(det["label"] for det in page["detections"]).most_common(4)
        top_s = ", ".join(f"{k}={v}" for k, v in top)
        lines.append(f"| {page['page']} | {len(page['detections'])} | {float(run['run_ms']):.1f} | {top_s} |")
    lines.extend([
        "",
        "| Label | count |",
        "|---|---:|",
    ])
    for label, count in labels.most_common():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=Path("assets/test3-2col-image.pdf"))
    ap.add_argument("--binary", type=Path, default=Path("build-metal/pp-doclayout-ggml"))
    ap.add_argument("--manifest", type=Path, default=Path("artifacts/v3_manifest.json"))
    ap.add_argument("--weights", type=Path, default=Path("artifacts/v3_weights.gguf"))
    ap.add_argument("--plan", type=Path, default=Path("artifacts/v3_plan.json"))
    ap.add_argument("--output-name", default="fetch_name_0")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workdir", type=Path, default=Path("artifacts/bench/pdf-warm-metal"))
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = list(manifest.get("labels", []))
    args.workdir.mkdir(parents=True, exist_ok=True)
    input_dir = args.workdir / "inputs"
    output_dir = args.workdir / "outputs"
    ann_dir = args.workdir / "annotated"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_annotate:
        ann_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    images = render_pdf(args.pdf, args.dpi)
    batch_rows = []
    for i, img in enumerate(images, 1):
        chw = preprocess(img, args.size)
        input_f32 = input_dir / f"page_{i:03d}.f32"
        output_f32 = output_dir / f"page_{i:03d}.f32"
        chw.tofile(input_f32)
        batch_rows.append((input_f32.resolve(), output_f32.resolve()))

    batch_list = args.workdir / "batch.tsv"
    batch_list.write_text("".join(f"{inp}\t{out}\n" for inp, out in batch_rows), encoding="utf-8")

    cmd = [
        str(args.binary),
        "--manifest", str(args.manifest),
        "--weights", str(args.weights),
        "--plan", str(args.plan),
        "--run-prefix", args.output_name,
        "--batch-f32-list", str(batch_list),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"warm batch failed:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}")
    cli = json.loads(res.stdout.strip().splitlines()[-1])

    pages = []
    label_counts: Counter[str] = Counter()
    for i, img in enumerate(images, 1):
        output_f32 = output_dir / f"page_{i:03d}.f32"
        rows = np.fromfile(output_f32, dtype=np.float32).reshape(-1, 7)
        detections = detections_from_output(rows, labels, args.threshold)
        label_counts.update(det["label"] for det in detections)
        page = {
            "page": i,
            "render_width": img.width,
            "render_height": img.height,
            "detections": detections,
        }
        pages.append(page)
        if not args.no_annotate:
            annotate(img, detections, args.size).save(ann_dir / f"page_{i:03d}.png")

    total_wall_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "pdf": str(args.pdf),
        "threshold": args.threshold,
        "cli": cli,
        "total_wall_ms": total_wall_ms,
        "pages": pages,
    }
    (args.workdir / "detections.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = markdown_report(args.pdf, cli, pages, label_counts, total_wall_ms)
    (args.workdir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.workdir / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
