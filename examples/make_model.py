"""生成用于演示/测试的 ONNX 模型。

模型结构（一个小型 CNN 分类头）:
    input (1,3,8,8)
      -> Conv(3->4, 3x3, stride=1, pad=1)
      -> Relu
      -> MaxPool(2x2)
      -> Flatten
      -> MatMul(64 -> 10)
      -> Add
      -> output (1,10)
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

MODEL_PATH = Path(__file__).resolve().parent / "test_model.onnx"


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    rng = np.random.default_rng(42)
    conv_w = helper.make_tensor(
        "conv.weight", TensorProto.FLOAT, [4, 3, 3, 3],
        rng.standard_normal((4, 3, 3, 3)).astype(np.float32).flatten(),
    )
    conv_b = helper.make_tensor(
        "conv.bias", TensorProto.FLOAT, [4],
        rng.standard_normal(4).astype(np.float32).flatten(),
    )
    fc_w = helper.make_tensor(
        "fc.weight", TensorProto.FLOAT, [64, 10],
        rng.standard_normal((64, 10)).astype(np.float32).flatten(),
    )
    fc_b = helper.make_tensor(
        "fc.bias", TensorProto.FLOAT, [10],
        rng.standard_normal(10).astype(np.float32).flatten(),
    )

    nodes = [
        helper.make_node(
            "Conv", ["input", "conv.weight", "conv.bias"], ["conv_out"],
            name="conv1", strides=[1, 1], pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1"),
        helper.make_node("MaxPool", ["relu_out"], ["pool_out"],
                         name="pool1", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["pool_out"], ["flat_out"], name="flat1", axis=1),
        helper.make_node("MatMul", ["flat_out", "fc.weight"], ["mm_out"], name="fc1"),
        helper.make_node("Add", ["mm_out", "fc.bias"], ["output"], name="bias_add"),
    ]

    model = helper.make_model(
        helper.make_graph(nodes, "demo_cnn", [x], [y],
                          initializer=[conv_w, conv_b, fc_w, fc_b]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # 兼容当前 onnxruntime 的最大 IR 版本
    model.ir_version = 9
    return model


def main():
    model = build_model()
    onnx.checker.check_model(model)
    MODEL_PATH.write_bytes(model.SerializeToString())
    print(f"测试模型已生成: {MODEL_PATH}")


if __name__ == "__main__":
    main()
