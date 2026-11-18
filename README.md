# EdgeAI Runtime —— 跨硬件边缘 AI 推理运行时（最小原型）

用一套代码加载 ONNX 模型并在不同后端（ONNX Runtime CPU / 模拟 NPU）上推理。
核心是一层**与硬件无关的统一中间表示（IR）+ 可插拔后端注册机制**。

## 项目结构

```
edge-ai-runtime/
├── runtime/                    # 运行时核心包
│   ├── __init__.py             #   对外 API 总出口（EdgeRuntime 等）
│   ├── api.py                  #   统一调用门面：load / run / 元信息，后端切换只改一个参数
│   ├── graph/
│   │   ├── model.py            #   统一 IR 数据结构：Node / GraphIR / TensorMeta
│   │   └── loader.py           #   ONNX -> IR 解析器（属性、权重、输入输出元信息）
│   ├── backend/
│   │   ├── base.py             #   后端抽象基类 BaseBackend：load/run 契约 + 算子白名单检查
│   │   ├── registry.py         #   后端注册表：@register_backend 注册，create_backend 实例化
│   │   ├── ort_cpu.py          #   后端 A：ONNX Runtime CPU（直接消费原始 ONNX 字节）
│   │   └── sim_npu.py          #   后端 B：模拟 NPU（编译->权重常驻->逐算子执行->profiling）
│   ├── ops/
│   │   └── numpy_ops.py        #   Numpy 算子库（Conv/MatMul/Relu/MaxPool 等），注册表模式
│   └── utils/
│       └── __init__.py         #   工具模块占位
├── examples/
│   ├── make_model.py           # 用 onnx.helper 生成测试模型（Conv->Relu->MaxPool->Flatten->MatMul->Add）
│   ├── demo.py                 # 演示：同一份代码在两个后端运行并做结果一致性校验
│   └── test_model.onnx         # （运行 make_model.py 后生成）
├── tests/
│   └── test_backends.py        # 端到端测试：加载/推理/两后端一致性/缺输入报错
└── requirements.txt            # numpy, onnx, onnxruntime
```

## 快速开始

```bash
pip install -r requirements.txt
python examples/make_model.py     # 生成测试模型
python examples/demo.py           # 双后端演示
python tests/test_backends.py     # 跑测试
```

## 使用方式（写一次，跑两端）

```python
import numpy as np
from runtime import EdgeRuntime

feed = {"input": np.random.randn(1, 3, 8, 8).astype(np.float32)}

rt = EdgeRuntime(backend="ort_cpu")   # 换成 "sim_npu" 即切换到模拟 NPU，"auto" 自动选择
rt.load("examples/test_model.onnx")
outputs = rt.run(**feed)
print(outputs["output"].shape)
```

## 扩展真实 NPU 后端（高通 QNN / 联发科 NeuroPilot）

1. 新建 `runtime/backend/qnn_npu.py`，继承 `BaseBackend`，实现 `_on_load`（对应 SDK 的模型转换/编译）与 `run`（SDK 推理调用）；
2. 加装饰器 `@register_backend("qnn_npu")`，依赖检查写入 `available` 类属性；
3. 在 `runtime/backend/__init__.py` 追加一行 import 触发注册；
4. 可选：把新后端加入 `api.py` 的 `AUTO_PREFERENCE` 列表，参与 auto 自动选择。

用户侧代码零改动。
