"""后端抽象基类 —— 跨硬件抽象层的核心契约。

任何新后端（未来接入高通 QNN、联发科 NeuroPilot 等）只需继承 BaseBackend
并实现 load / run 两个方法，再用 @register_backend 注册，即可无缝接入
统一的 EdgeRuntime 调用接口，用户侧代码零改动。
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ..graph.model import GraphIR


class BackendError(Exception):
    """后端相关错误基类。"""


class UnsupportedOpError(BackendError):
    """图中有后端不支持的算子（真实 NPU 场景需要 CPU fallback）。"""

    def __init__(self, backend_name: str, ops: List[str]):
        self.ops = ops
        super().__init__(
            f"后端 '{backend_name}' 不支持以下算子: {', '.join(sorted(set(ops)))}"
        )


class BaseBackend(ABC):
    """所有后端必须实现的统一契约。"""

    name: str = "base"             # 注册名，如 "ort_cpu" / "sim_npu"
    description: str = ""          # 人类可读描述
    available: bool = True         # 依赖是否满足（注册表据此做 auto 选择与降级）

    def __init__(self):
        self._ir: GraphIR | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def load(self, ir: GraphIR) -> None:
        """加载模型（IR）。子类可覆写，做编译/部署等硬件相关工作。

        默认实现会先做算子支持性检查（后端可声明 skip_op_check 跳过）。
        """
        if not getattr(self, "skip_op_check", False):
            missing = self.check_ops(ir)
            if missing:
                raise UnsupportedOpError(self.name, missing)
        self._ir = ir
        self._on_load(ir)

    def _on_load(self, ir: GraphIR) -> None:
        """钩子方法：子类真正的加载逻辑放这里。"""

    # ------------------------------------------------------------------
    # 必须实现的接口
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """同步推理：输入张量名 -> 数组，返回 输出张量名 -> 数组。"""

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def input_names(self) -> List[str]:
        return self._ir.inputs if self._ir else []

    @property
    def output_names(self) -> List[str]:
        return self._ir.outputs if self._ir else []

    def check_ops(self, ir: GraphIR) -> List[str]:
        """返回图中本后端不支持的算子列表。默认：不做检查。"""
        return []

    def info(self) -> str:
        return f"[{self.name}] {self.description}"
