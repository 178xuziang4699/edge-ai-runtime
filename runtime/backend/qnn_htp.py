"""Qualcomm QNN HTP 后端插件 —— ctypes 接入骨架（需要真机 + QNN SDK 才能完整运行）。

前提：
1. 设备上存在 libQnnHtp.so 及配套 libQnnHtpVXXStub.so / VXXSkel.so
2. 已用 tools/convert_qnn.py 离线生成 context binary（.serialized.bin）

实现说明：QNN 的所有函数经 QnnInterface_t 函数指针表(vtable)访问，
各字段与结构体布局以所装 SDK 的 include/QnnInterface.h、QnnBackend.h、
QnnContext.h、QnnGraph.h 为准。本骨架保持与官方 qnn-sample-app 相同的
调用顺序：getProviders -> backend_create -> context_createFromBinary ->
graph_retrieve -> execute。
"""

import ctypes as C
import os
from pathlib import Path

from ..backend.base import BaseBackend, BackendError
from ..backend.registry import register_backend

QNN_LIB_CANDIDATES = [
    "/usr/lib/libQnnHtp.so",
    "/vendor/lib64/libQnnHtp.so",            # Android
    "/system/vendor/lib64/libQnnHtp.so",     # Android
]


def find_qnn_lib() -> str | None:
    for p in QNN_LIB_CANDIDATES:
        if Path(p).exists():
            return p
    return None


@register_backend("qnn_htp")
class QnnHtpBackend(BaseBackend):
    name = "qnn_htp"
    description = "Qualcomm QNN Hexagon NPU 后端（context binary + ctypes）"
    available = find_qnn_lib() is not None

    # 白名单覆盖矩阵（算子 + 属性约束），见 tools/coverage/qnn_htp.yaml
    SUPPORTED_OPS = {"Conv", "MatMul", "Gemm", "Relu", "Add", "Mul",
                     "MaxPool", "AveragePool", "Flatten", "Reshape",
                     "Softmax", "Sigmoid", "Concat", "Split"}

    def __init__(self, model_file: str | None = None, soc_arch: str = "v73"):
        super().__init__()
        self.model_file = model_file      # compile() 产出的 .serialized.bin
        self.soc_arch = soc_arch
        self._lib = None
        self._if = None                   # QNN 函数指针表（SDK vtable）
        self._backend = self._context = self._graph = None

    # ---------------- 离线编译（在装有 SDK 工具链的主机上运行） ----------------
    @classmethod
    def compile(cls, onnx_path: str, out_dir: str = "artifacts", soc_arch: str = "v73") -> str:
        """调用 tools/convert_qnn.py 生成 context binary，返回其路径。"""
        import subprocess
        script = Path(__file__).resolve().parents[2] / "tools" / "convert_qnn.py"
        result = subprocess.run(
            ["python3", str(script), onnx_path,
             "--soc-arch", soc_arch, "--sdk-bin", os.environ.get("QNN_SDK_BIN", "")],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip().split(": ")[-1]

    # ---------------- 设备端加载 ----------------
    def load_model_file(self, model_file: str) -> None:
        """QNN 无法从 IR 在设备端编译，必须加载离线产出的 context binary。"""
        if not self.available:
            raise BackendError("未找到 libQnnHtp.so，QNN 后端不可用")
        self.model_file = model_file
        self._lib = C.CDLL(find_qnn_lib())

        # TODO(真机适配): 以下顺序对应 qnn-sample-app，函数指针取自 QnnInterface_t：
        #   1. QnnInterface_getProviders  -> 取 HTP provider 的函数表
        #   2. backend_create(device_id, log, &backend)
        #   3. device_create(backend, device_cfg, &device)
        #   4. context_create_from_binary(backend, device, binary_info, &context)
        #   5. graph_retrieve(context, graph_name, &graph)
        #   6. graph_get_info(graph) -> 输入输出张量的名称/shape/dtype/量化参数
        raise NotImplementedError("按所装 SDK 头文件补齐 vtable 定义后启用")

    # ---------------- 与 BaseBackend 契约的对接 ----------------
    def _on_load(self, ir):
        # 运行时接口：QNN 走 context binary，忽略 IR 本体，仅用它做白名单校验
        if self.model_file is None:
            raise BackendError(
                "QNN 后端需要离线转换产物，请先运行 QnnHtpBackend.compile()"
            )
        self.load_model_file(self.model_file)

    def run(self, feed):
        # TODO(真机适配):
        #   1. 校验输入 shape/dtype（QNN 严格静态 shape）
        #   2. 将 numpy 数组写入预注册的 Qnn_Tensor_t buffer
        #   3. graph_execute(graph, signal, ctx)  # 同步执行
        #   4. 读输出 buffer -> numpy dict
        raise NotImplementedError

    def check_ops(self, ir):
        return [n.op_type for n in ir.nodes if n.op_type not in self.SUPPORTED_OPS]

    # ---------------- 资源释放（顺序与创建相反） ----------------
    def __del__(self):
        # TODO: graph free -> context free -> device free -> backend free
        pass
