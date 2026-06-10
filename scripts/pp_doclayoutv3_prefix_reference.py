#!/usr/bin/env python3
"""Generate deterministic ONNXRuntime reference tensors for PP-DocLayoutV3 prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def require_onnx():
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise SystemExit("missing onnx; install or add it to PYTHONPATH") from exc
    return onnx


def require_ort():
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise SystemExit("missing onnxruntime; install or add it to PYTHONPATH") from exc
    return ort


def value_info_by_name(onnx, model, name: str):
    inferred = onnx.shape_inference.infer_shapes(model)
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.name == name:
            return value
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-name", default="p2o.pd_op.relu.0.0")
    parser.add_argument("--input-out", required=True, type=Path)
    parser.add_argument("--reference-out", required=True, type=Path)
    parser.add_argument("--shape-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    onnx = require_onnx()
    ort = require_ort()
    model = onnx.load(str(args.onnx), load_external_data=False)
    del model.graph.output[:]
    model.graph.output.extend([value_info_by_name(onnx, model, args.output_name)])

    rng = np.random.default_rng(args.seed)
    image = rng.normal(0.0, 0.25, size=(1, 3, 800, 800)).astype(np.float32)
    im_shape = np.array([[800.0, 800.0]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    output = session.run(
        [args.output_name],
        {
            "image": image,
            "im_shape": im_shape,
            "scale_factor": scale_factor,
        },
    )[0]

    args.input_out.parent.mkdir(parents=True, exist_ok=True)
    args.reference_out.parent.mkdir(parents=True, exist_ok=True)
    args.shape_out.parent.mkdir(parents=True, exist_ok=True)
    image.tofile(args.input_out)
    output.astype(np.float32, copy=False).tofile(args.reference_out)
    args.shape_out.write_text(
        f"input_shape={list(image.shape)}\n"
        f"output_name={args.output_name}\n"
        f"output_shape={list(output.shape)}\n",
        encoding="utf-8",
    )
    print(f"wrote {args.input_out} and {args.reference_out}; output_shape={list(output.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
