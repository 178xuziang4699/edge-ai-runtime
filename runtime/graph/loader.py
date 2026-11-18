"""ONNX 模型加载器：把 .onnx 文件解析为统一的 GraphIR。"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from .model import GraphIR, Node, TensorMeta


def _parse_attrs(attrs) -> Dict[str, Any]:
    """把 onnx.AttributeProto 列表解析成 Python 原生 dict。"""
    result: Dict[str, Any] = {}
    for a in attrs:
        if a.type == onnx.AttributeProto.FLOAT:
            result[a.name] = float(a.f)
        elif a.type == onnx.AttributeProto.INT:
            result[a.name] = int(a.i)
        elif a.type == onnx.AttributeProto.STRING:
            result[a.name] = a.s.decode("utf-8")
        elif a.type == onnx.AttributeProto.INTS:
            result[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            result[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.TENSOR:
            result[a.name] = numpy_helper.to_array(a.t)
        else:
            # 未识别的属性类型直接透传原始 proto，避免静默丢失
            result[a.name] = a
    return result


def load_onnx(path) -> Tuple[bytes, GraphIR]:
    """加载 ONNX 文件。

    Returns:
        (raw_bytes, GraphIR):
        - raw_bytes: 原始序列化模型，供直接消费 onnx 的后端（如 ORT）使用
        - GraphIR:   统一中间表示，供需要自己执行图的 NPU 类后端使用
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    model = onnx.load(str(path))
    graph = model.graph

    nodes = [
        Node(
            name=n.name or f"{n.op_type}_{idx}",
            op_type=n.op_type,
            inputs=list(n.input),
            outputs=list(n.output),
            attrs=_parse_attrs(n.attribute),
        )
        for idx, n in enumerate(graph.node)
    ]

    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    # 模型外部输入 = graph.input 里排除 initializer（ONNX 允许权重同时出现在 input 中）
    inputs = [i.name for i in graph.input if i.name not in initializers]
    outputs = [o.name for o in graph.output]

    tensor_metas: Dict[str, TensorMeta] = {}
    for info in list(graph.input) + list(graph.output):
        shape = [None if d.dim_param else d.dim_value for d in info.type.tensor_type.shape.dim]
        tensor_metas[info.name] = TensorMeta(
            name=info.name,
            dtype=onnx.TensorProto.DataType.Name(info.type.tensor_type.elem_type).lower(),
            shape=shape,
        )

    ir = GraphIR(
        name=graph.name or path.stem,
        nodes=nodes,
        initializers=initializers,
        inputs=inputs,
        outputs=outputs,
        tensor_metas=tensor_metas,
        opset=model.opset_import[0].version if model.opset_import else 13,
        source_bytes=model.SerializeToString(),
    )
    return ir.source_bytes, ir
