"""深穹 EdgeAI Runtime —— 跨硬件边缘 AI 推理运行时（最小原型）。

统一调用入口:
    from runtime import EdgeRuntime
    rt = EdgeRuntime(backend="sim_npu")   # 或 "ort_cpu" / "auto"
    rt.load("model.onnx")
    outputs = rt.run(input_x=np_array)
"""

from .api import EdgeRuntime, create_runtime
from .graph.model import GraphIR, Node, TensorMeta
from .backend.registry import (
    list_backends,
    register_backend,
    create_backend,
)

__version__ = "0.1.0"

__all__ = [
    "EdgeRuntime",
    "create_runtime",
    "GraphIR",
    "Node",
    "TensorMeta",
    "list_backends",
    "register_backend",
    "create_backend",
]
