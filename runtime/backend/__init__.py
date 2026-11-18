from .base import BaseBackend, BackendError, UnsupportedOpError
from .registry import register_backend, create_backend, list_backends, _BACKENDS

# import 触发各后端模块的注册（新增后端后在此追加一行 import 即可）
from . import ort_cpu  # noqa: F401
from . import sim_npu  # noqa: F401
from . import qnn_htp  # noqa: F401  # 真机依赖缺失时 available=False，auto 自动跳过

__all__ = [
    "BaseBackend",
    "BackendError",
    "UnsupportedOpError",
    "register_backend",
    "create_backend",
    "list_backends",
]
