"""统一中间表示（IR）。

所有后端（ONNX Runtime / 模拟 NPU / 未来真实 NPU）都面对同一套 IR 结构，
后端可以自由选择：直接消费 IR 执行，或基于 IR 做图优化后再下发给硬件。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class Node:
    """图中的一个算子节点，与 ONNX NodeProto 一一对应。"""

    name: str                 # 节点名（ONNX 中可能为空，会自动补全）
    op_type: str              # 算子类型，如 "Conv" / "MatMul" / "ReLU"
    inputs: List[str]         # 输入张量名
    outputs: List[str]        # 输出张量名
    attrs: Dict[str, Any] = field(default_factory=dict)  # 已解析为 Python 原生类型

    def __repr__(self) -> str:
        return f"Node({self.name}: {self.op_type} {self.inputs} -> {self.outputs})"


@dataclass
class TensorMeta:
    """模型输入/输出张量的元信息。"""

    name: str
    dtype: str = "float32"        # numpy dtype 字符串
    shape: Sequence[Any] = ()     # 含动态维度时用 None 占位


@dataclass
class GraphIR:
    """与硬件无关的计算图表示。"""

    name: str
    nodes: List[Node]                 # 拓扑有序（ONNX 规范保证）
    initializers: Dict[str, np.ndarray]   # 常量权重: 张量名 -> 数组
    inputs: List[str]                 # 模型外部输入名（排除 initializer）
    outputs: List[str]                # 模型输出名
    tensor_metas: Dict[str, TensorMeta] = field(default_factory=dict)
    opset: int = 13
    source_bytes: bytes = b""         # 原始 onnx 序列化字节，供直接消费 onnx 的后端使用

    def summary(self) -> str:
        """返回一张人类可读的图摘要（调试用）。"""
        lines = [f"Graph '{self.name}' (opset={self.opset})"]
        op_count: Dict[str, int] = {}
        for n in self.nodes:
            op_count[n.op_type] = op_count.get(n.op_type, 0) + 1
        lines.append("  算子统计: " + ", ".join(f"{k}x{v}" for k, v in sorted(op_count.items())))
        lines.append(f"  输入: {self.inputs}")
        lines.append(f"  输出: {self.outputs}")
        for n in self.nodes:
            lines.append(f"  [{n.op_type}] {n.name}: {n.inputs} -> {n.outputs}")
        return "\n".join(lines)
