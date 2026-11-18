"""统一 API：用户唯一需要接触的入口。

写一次代码，切换后端只需改一个参数：
    rt = EdgeRuntime(backend="ort_cpu")   # 跑 ONNX Runtime
    rt = EdgeRuntime(backend="sim_npu")   # 跑模拟 NPU
"""

from typing import Dict, List, Optional

import numpy as np

from .backend.base import BaseBackend
from .backend.registry import _BACKENDS, create_backend
from .graph.loader import load_onnx
from .graph.model import GraphIR

# auto 模式下的后端优先级（未来接入真实 NPU 后，把硬件后端排在前面即可）
AUTO_PREFERENCE = ["ort_cpu", "sim_npu"]


def _resolve_backend(name: str) -> BaseBackend:
    if name != "auto":
        return create_backend(name)
    for cand in AUTO_PREFERENCE:
        if cand in _BACKENDS and _BACKENDS[cand]().available:
            return create_backend(cand)
    raise RuntimeError("auto 模式下没有任何可用后端")


class EdgeRuntime:
    """边缘 AI 推理运行时 —— 统一调用门面（Facade）。"""

    def __init__(self, backend: str = "auto", verbose: bool = True):
        self._backend = _resolve_backend(backend)
        self.backend_name = self._backend.name  # 实际选中的后端名（auto 解析后）
        self._ir: Optional[GraphIR] = None
        self._verbose = verbose
        if verbose:
            print(f"[EdgeRuntime] 当前后端: {self._backend.info()}")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self, model_path: str) -> GraphIR:
        """加载 ONNX 模型并下发给当前后端（真实 NPU 场景此处对应模型转换/部署）。"""
        raw_bytes, ir = load_onnx(model_path)
        self._backend.load(ir)
        self._ir = ir
        if self._verbose:
            print(f"[EdgeRuntime] 模型已加载: {model_path}")
        return ir

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def run(self, **feed: np.ndarray) -> Dict[str, np.ndarray]:
        """同步推理。键为模型输入名，值为 numpy 数组。"""
        if self._ir is None:
            raise RuntimeError("请先调用 load() 加载模型")
        return self._backend.run(feed)

    # 兼容 dict 风格调用
    def run_dict(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.run(**feed)

    # ------------------------------------------------------------------
    # 元信息透传
    # ------------------------------------------------------------------
    @property
    def input_names(self) -> List[str]:
        return self._backend.input_names

    @property
    def output_names(self) -> List[str]:
        return self._backend.output_names

    def graph_summary(self) -> str:
        if self._ir is None:
            return "(尚未加载模型)"
        return self._ir.summary()

    def backend(self) -> BaseBackend:
        return self._backend


def create_runtime(backend: str = "auto", verbose: bool = True) -> EdgeRuntime:
    """工厂函数，语义化创建运行时实例。"""
    return EdgeRuntime(backend=backend, verbose=verbose)
