"""后端 A：ONNX Runtime（CPU）。

直接消费原始 ONNX 字节流，把执行完全交给 ORT 引擎。
代表"成熟商业推理引擎接入"的路径。
"""

from typing import Dict

import numpy as np

from ..graph.model import GraphIR
from .base import BaseBackend
from .registry import register_backend


def _ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@register_backend("ort_cpu")
class OnnxRuntimeCPUBackend(BaseBackend):
    name = "ort_cpu"
    description = "ONNX Runtime CPU 后端（CPUExecutionProvider）"
    # ORT 直接消费原始模型字节，无需 IR 算子白名单检查
    skip_op_check = True
    available = _ort_available()

    def _on_load(self, ir: GraphIR) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            ir.source_bytes, providers=["CPUExecutionProvider"]
        )
        # 以 ORT 实际会话的输入输出为准（更权威）
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

    def run(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        outputs = self._session.run(self._output_names, feed)
        return dict(zip(self._output_names, outputs))

    @property
    def input_names(self):
        return getattr(self, "_input_names", super().input_names)

    @property
    def output_names(self):
        return getattr(self, "_output_names", super().output_names)
