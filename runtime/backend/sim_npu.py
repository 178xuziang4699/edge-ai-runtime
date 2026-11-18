"""后端 B：模拟 NPU 后端（纯 Numpy 实现）。

模拟真实 NPU 的软件栈行为：
    1. "编译"阶段 —— 校验算子白名单（对应 NPU SDK 的算子支持列表）
    2. "部署"阶段 —— 把权重常驻内存（对应权重搬运到 NPU 片上内存）
    3. "推理"阶段 —— 按图拓扑逐算子执行，并统计各算子耗时

未来接入真实 NPU（高通 QNN / 联发科 NeuroPilot）时，
把 numpy 算子替换为对应 SDK 的算子调用即可，整体流程不变。
"""

import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..graph.model import GraphIR, Node
from ..ops.numpy_ops import OP_REGISTRY
from .base import BaseBackend, UnsupportedOpError
from .registry import register_backend


@register_backend("sim_npu")
class SimulatedNPUBackend(BaseBackend):
    name = "sim_npu"
    description = "模拟 NPU 后端（Numpy 实现 Conv/MatMul/Relu 等常见算子）"

    def __init__(self):
        super().__init__()
        self._weights: Dict[str, np.ndarray] = {}
        self._op_stats: Dict[str, List[float]] = defaultdict(list)
        self._verbose = True

    # ---------------- 算子白名单检查（模拟 NPU SDK 的 compile 阶段） ----------------
    def check_ops(self, ir: GraphIR) -> List[str]:
        return [n.op_type for n in ir.nodes if n.op_type not in OP_REGISTRY]

    # ---------------- "编译 + 权重搬运" ----------------
    def _on_load(self, ir: GraphIR) -> None:
        unsupported = self.check_ops(ir)
        if unsupported:
            raise UnsupportedOpError(self.name, unsupported)

        self._weights = {k: v.copy() for k, v in ir.initializers.items()}
        if self._verbose:
            total_kb = sum(w.nbytes for w in self._weights.values()) / 1024
            ops = sorted({n.op_type for n in ir.nodes})
            print(
                f"[sim_npu] 模型编译完成: {len(ir.nodes)} 个算子, "
                f"权重常驻 {total_kb:.1f} KB, 算子集: {ops}"
            )

    # ---------------- "推理" ----------------
    def run(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._ir is None:
            raise RuntimeError("模型尚未加载，请先调用 runtime.load()")
        ir = self._ir

        # 校验输入完整性
        missing = [n for n in ir.inputs if n not in feed]
        if missing:
            raise KeyError(f"缺少输入张量: {missing}（需要: {ir.inputs}）")

        # 工作区: 常量权重 + 外部输入
        workspace: Dict[str, np.ndarray] = dict(self._weights)
        workspace.update({k: np.asarray(v, dtype=np.float32) for k, v in feed.items()})

        for node in ir.nodes:  # ONNX 保证拓扑有序
            t0 = time.perf_counter()
            node_inputs = [workspace[name] for name in node.inputs]
            op_fn = OP_REGISTRY[node.op_type]
            (result,) = op_fn(node, node_inputs)  # 本原型算子均为单输出
            workspace[node.outputs[0]] = result
            self._op_stats[node.op_type].append((time.perf_counter() - t0) * 1000)

        results = {name: workspace[name] for name in ir.outputs}
        if self._verbose:
            stats = ", ".join(
                f"{op}={sum(ts) / len(ts):.2f}ms" for op, ts in self._op_stats.items()
            )
            print(f"[sim_npu] 推理完成 ({stats})")
        return results

    # ---------------- 观测接口（真实 NPU 也会提供 profiling 能力） ----------------
    def profile(self) -> Dict[str, float]:
        """返回每个算子的累计耗时（毫秒）。"""
        return {op: sum(ts) for op, ts in self._op_stats.items()}
