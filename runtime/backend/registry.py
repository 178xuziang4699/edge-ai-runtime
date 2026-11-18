"""后端注册表：名称 <-> 后端类 的映射，支撑可插拔的多硬件后端体系。

新增后端三步走：
    1. 在 runtime/backend/ 下新建模块（如 qnn_npu.py）
    2. 实现类继承 BaseBackend，并加 @register_backend("qnn_npu")
    3. 在 runtime/backend/__init__.py 中 import 该模块（触发注册）
"""

from typing import Dict, Type

from .base import BaseBackend

_BACKENDS: Dict[str, Type[BaseBackend]] = {}


def register_backend(name: str):
    """后端注册装饰器。"""

    def deco(cls: Type[BaseBackend]) -> Type[BaseBackend]:
        if name in _BACKENDS:
            raise ValueError(f"后端重复注册: {name}")
        cls.name = name
        _BACKENDS[name] = cls
        return cls

    return deco


def create_backend(name: str) -> BaseBackend:
    """按名称实例化后端；依赖缺失时给出明确错误。"""
    if name not in _BACKENDS:
        raise KeyError(
            f"未注册的后端 '{name}'，可用后端: {list(_BACKENDS)}"
        )
    backend = _BACKENDS[name]()
    if not backend.available:
        raise RuntimeError(
            f"后端 '{name}' 当前不可用（请检查依赖是否安装，"
            f"例如 pip install onnxruntime）"
        )
    return backend


def list_backends() -> Dict[str, dict]:
    """列出全部已注册后端及其可用性。"""
    return {
        name: {
            "description": cls.description,
            "available": cls().available if cls().available is not None else False,
        }
        for name, cls in _BACKENDS.items()
    }
