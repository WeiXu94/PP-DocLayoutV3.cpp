#!/usr/bin/env python3
"""Create a PP-DocLayoutV3 graph manifest for the native GGML runner.

This does not convert the full detector to GGUF yet. It freezes the ONNX graph
contract and operator inventory so the C++ runner can be implemented and tested
against one explicit subset at a time.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27

ONNX_TO_GGML_TYPE = {
    1: GGML_TYPE_F32,
    6: GGML_TYPE_I32,
    7: GGML_TYPE_I64,
}


def require_onnx():
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "missing onnx. For local inspection, use the bundled runtime, e.g.:\n"
            "  PYTHONPATH=/tmp/onnx-inspect-py312 "
            "/Users/weixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 "
            "scripts/pp_doclayoutv3_onnx_to_ggml_manifest.py --onnx ... --config ... --out ..."
        ) from exc
    return onnx


def tensor_shape(value_info) -> list[int | str]:
    shape = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def tensor_summary(value_info) -> dict:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elem_type": int(tensor_type.elem_type),
        "shape": tensor_shape(value_info),
    }


def shape_map(model) -> dict[str, dict]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    return {value.name: tensor_summary(value) for value in values}


def initializer_summary(tensor) -> dict:
    parameter_count = 1
    for dim in tensor.dims:
        parameter_count *= int(dim)
    return {
        "name": tensor.name,
        "data_type": int(tensor.data_type),
        "shape": [int(dim) for dim in tensor.dims],
        "parameter_count": parameter_count,
    }


def ggml_shape_from_onnx(dims) -> list[int]:
    return [int(dim) for dim in reversed(dims)]


def tensor_data_bytes(onnx, tensor) -> bytes:
    array = onnx.numpy_helper.to_array(tensor)
    return array.tobytes(order="C")


def pack_u32(value: int) -> bytes:
    return struct.pack("<I", value)


def pack_u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def pack_i64(value: int) -> bytes:
    return struct.pack("<q", value)


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return pack_u64(len(raw)) + raw


def pad_len(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return (alignment - (offset % alignment)) % alignment


def write_kv_string(out, key: str, value: str) -> None:
    out.write(pack_string(key))
    out.write(pack_u32(GGUF_TYPE_STRING))
    out.write(pack_string(value))


def write_kv_u32(out, key: str, value: int) -> None:
    out.write(pack_string(key))
    out.write(pack_u32(GGUF_TYPE_UINT32))
    out.write(pack_u32(value))


def write_kv_string_array(out, key: str, values: list[str]) -> None:
    out.write(pack_string(key))
    out.write(pack_u32(GGUF_TYPE_ARRAY))
    out.write(pack_u32(GGUF_TYPE_STRING))
    out.write(pack_u64(len(values)))
    for value in values:
        out.write(pack_string(value))


def build_gguf_tensor_records(onnx, graph) -> tuple[list[dict], int]:
    records = []
    offset = 0
    total_bytes = 0
    for tensor in graph.initializer:
        if tensor.data_type not in ONNX_TO_GGML_TYPE:
            raise SystemExit(
                f"unsupported initializer data type {tensor.data_type}: {tensor.name}"
            )
        data = tensor_data_bytes(onnx, tensor)
        records.append(
            {
                "name": tensor.name,
                "dims": ggml_shape_from_onnx(tensor.dims),
                "ggml_type": ONNX_TO_GGML_TYPE[tensor.data_type],
                "offset": offset,
                "data": data,
            }
        )
        total_bytes += len(data)
        offset += len(data) + pad_len(len(data))
    return records, total_bytes


def write_gguf_weights(
    onnx,
    graph,
    config: dict,
    onnx_path: Path,
    config_path: Path,
    out_path: Path,
) -> dict:
    records, total_weight_bytes = build_gguf_tensor_records(onnx, graph)
    kv_count = 8
    labels = config.get("label_list", [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        out.write(b"GGUF")
        out.write(pack_u32(GGUF_VERSION))
        out.write(pack_i64(len(records)))
        out.write(pack_i64(kv_count))

        write_kv_string(out, "general.architecture", "pp-doclayoutv3")
        write_kv_string(
            out,
            "general.name",
            config.get("Global", {}).get("model_name", "PP-DocLayoutV3"),
        )
        write_kv_u32(out, "general.alignment", GGUF_DEFAULT_ALIGNMENT)
        write_kv_string(out, "archon.source.onnx_path", str(onnx_path))
        write_kv_string(out, "archon.source.config_path", str(config_path))
        write_kv_u32(out, "archon.pp_doclayout.input_size", 800)
        write_kv_string(out, "archon.pp_doclayout.layout", "NCHW")
        write_kv_string_array(out, "archon.pp_doclayout.labels", labels)

        for record in records:
            out.write(pack_string(record["name"]))
            out.write(pack_u32(len(record["dims"])))
            for dim in record["dims"]:
                out.write(pack_u64(dim))
            out.write(pack_u32(record["ggml_type"]))
            out.write(pack_u64(record["offset"]))

        out.write(b"\0" * pad_len(out.tell()))

        for record in records:
            out.write(record["data"])
            out.write(b"\0" * pad_len(len(record["data"])))

    return {
        "path": str(out_path),
        "tensor_count": len(records),
        "weight_bytes": total_weight_bytes,
        "file_bytes": out_path.stat().st_size,
    }


def jsonable_attr_value(onnx, value):
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [jsonable_attr_value(onnx, item) for item in value]
    if isinstance(value, onnx.TensorProto):
        return {
            "tensor_name": value.name,
            "data_type": int(value.data_type),
            "shape": [int(dim) for dim in value.dims],
        }
    if isinstance(value, onnx.GraphProto):
        return {"graph_name": value.name, "node_count": len(value.node)}
    return str(value)


def node_attributes(onnx, node) -> dict:
    attrs = {}
    for attr in node.attribute:
        attrs[attr.name] = jsonable_attr_value(onnx, onnx.helper.get_attribute_value(attr))
    return attrs


def build_execution_plan(onnx, model, config: dict, gguf_summary: dict | None) -> dict:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = shape_map(inferred)
    initializer_names = {tensor.name for tensor in inferred.graph.initializer}
    nodes = []
    for index, node in enumerate(inferred.graph.node):
        nodes.append(
            {
                "index": index,
                "op_type": node.op_type,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": node_attributes(onnx, node),
            }
        )

    return {
        "format": "archon.pp-doclayoutv3.execution-plan",
        "format_version": 1,
        "model_name": config.get("Global", {}).get("model_name", "PP-DocLayoutV3"),
        "preprocess": preprocess_from_config(config),
        "labels": config.get("label_list", []),
        "inputs": [tensor_summary(value_info) for value_info in inferred.graph.input],
        "outputs": [tensor_summary(value_info) for value_info in inferred.graph.output],
        "node_count": len(nodes),
        "initializer_count": len(initializer_names),
        "gguf_weights": gguf_summary,
        "value_shapes": shapes,
        "initializer_names": sorted(initializer_names),
        "nodes": nodes,
    }


def preprocess_from_config(config: dict) -> dict:
    steps = config.get("Preprocess") or []
    resize = next((step for step in steps if step.get("type") == "Resize"), {})
    normalize = next((step for step in steps if step.get("type") == "NormalizeImage"), {})
    return {
        "target_size": resize.get("target_size", [800, 800]),
        "keep_ratio": bool(resize.get("keep_ratio", False)),
        "interp": resize.get("interp", 2),
        "mean": normalize.get("mean", [0.0, 0.0, 0.0]),
        "std": normalize.get("std", [1.0, 1.0, 1.0]),
        "norm_type": normalize.get("norm_type", "none"),
        "layout": "NCHW",
    }


def build_manifest(onnx_path: Path, config_path: Path, gguf_summary: dict | None) -> dict:
    onnx = require_onnx()
    model = onnx.load(str(onnx_path), load_external_data=False)
    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    graph = model.graph
    op_histogram = collections.Counter(node.op_type for node in graph.node)
    initializers = [initializer_summary(tensor) for tensor in graph.initializer]
    parameter_count = sum(item["parameter_count"] for item in initializers)

    manifest = {
        "format": "archon.pp-doclayoutv3.ggml-manifest",
        "format_version": 1,
        "model_name": config.get("Global", {}).get("model_name", "PP-DocLayoutV3"),
        "source": {
            "onnx_path": str(onnx_path),
            "config_path": str(config_path),
            "opsets": [
                {"domain": opset.domain, "version": int(opset.version)}
                for opset in model.opset_import
            ],
            "ir_version": int(model.ir_version),
        },
        "preprocess": preprocess_from_config(config),
        "labels": config.get("label_list", []),
        "inputs": [tensor_summary(value_info) for value_info in graph.input],
        "outputs": [tensor_summary(value_info) for value_info in graph.output],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "parameter_count": parameter_count,
        "op_histogram": dict(sorted(op_histogram.items())),
        "unsupported_ops": sorted(op_histogram.keys()),
        "initializers_largest": sorted(
            initializers,
            key=lambda item: item["parameter_count"],
            reverse=True,
        )[:50],
    }
    if gguf_summary is not None:
        manifest["gguf_weights"] = gguf_summary
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gguf-out", type=Path)
    parser.add_argument("--plan-out", type=Path)
    args = parser.parse_args()

    onnx = None
    model = None
    config = None
    gguf_summary = None
    if args.gguf_out is not None or args.plan_out is not None:
        onnx = require_onnx()
        model = onnx.load(str(args.onnx), load_external_data=False)
        with args.config.open(encoding="utf-8") as fh:
            config = json.load(fh)
    if args.gguf_out is not None:
        gguf_summary = write_gguf_weights(
            onnx,
            model.graph,
            config,
            args.onnx,
            args.config,
            args.gguf_out,
        )

    manifest = build_manifest(args.onnx, args.config, gguf_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.plan_out is not None:
        plan = build_execution_plan(onnx, model, config, gguf_summary)
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} with {manifest['node_count']} nodes, "
        f"{manifest['initializer_count']} initializers, "
        f"{len(manifest['op_histogram'])} op families"
    )
    if gguf_summary is not None:
        print(
            f"wrote {gguf_summary['path']} with {gguf_summary['tensor_count']} tensors, "
            f"{gguf_summary['weight_bytes']} raw weight bytes"
        )
    if args.plan_out is not None:
        print(f"wrote {args.plan_out} execution plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
