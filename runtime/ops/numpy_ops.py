"""模拟 NPU 后端使用的 Numpy 算子库。

采用注册表模式：新增算子只需写一个函数并加 @register_op("OpName") 装饰器，
模拟 NPU 后端执行图时会自动查表调用 —— 这也是未来接入真实 NPU 时
"算子 -> 硬件指令" 映射层的雏形。

支持的算子集合与真实 NPU SDK 的"算子白名单"概念一致：
不在白名单内的算子需要 fallback 到 CPU（本原型中直接抛错提示）。
"""

from typing import Callable, Dict, List

import numpy as np

# 算子注册表: op_type -> callable(node, input_arrays) -> list[output_arrays]
OP_REGISTRY: Dict[str, Callable] = {}


def register_op(op_type: str):
    """算子注册装饰器。

    算子函数签名统一为:
        fn(node, inputs: List[np.ndarray]) -> List[np.ndarray]
    输入输出均为数组列表（与 ONNX 节点多输入/多输出的语义对齐）。
    """

    def deco(fn: Callable) -> Callable:
        if op_type in OP_REGISTRY:
            raise ValueError(f"算子重复注册: {op_type}")
        OP_REGISTRY[op_type] = fn
        return fn

    return deco


def list_supported_ops() -> List[str]:
    return sorted(OP_REGISTRY)


# ---------------------------------------------------------------------------
# 算子实现
# ---------------------------------------------------------------------------

@register_op("Conv")
def op_conv(node, inputs):
    """2D 卷积（NCHW 布局），支持 stride / pad / dilation / group。"""
    x, w = inputs[0], inputs[1]
    b = inputs[2] if len(inputs) > 2 else None

    strides = node.attrs.get("strides", [1, 1])
    pads = node.attrs.get("pads", [0, 0, 0, 0])          # ONNX: [t, l, b, r]
    dilations = node.attrs.get("dilations", [1, 1])
    group = node.attrs.get("group", 1)

    n, c, h, wd = x.shape
    m, c_g, kh, kw = w.shape
    sh, sw = strides
    pt, pl, pb, pr = pads
    dh, dw = dilations

    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    ho = (xp.shape[2] - (dh * (kh - 1) + 1)) // sh + 1
    wo = (xp.shape[3] - (dw * (kw - 1) + 1)) // sw + 1

    out = np.empty((n, m, ho, wo), dtype=np.float32)
    c_per_g = c // group
    m_per_g = m // group

    # im2col：按卷积核位置滑动切片，向量化计算
    for g in range(group):
        wg = w[g * m_per_g : (g + 1) * m_per_g]                     # (m_g, c_g, kh, kw)
        cols = np.empty((n, c_g, kh, kw, ho, wo), dtype=np.float32)
        xg = xp[:, g * c_per_g : (g + 1) * c_per_g]
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j] = xg[
                    :,
                    :,
                    i * dh : i * dh + ho * sh : sh,
                    j * dw : j * dw + wo * sw : sw,
                    ]
        cols = cols.reshape(n, c_g * kh * kw, ho * wo)
        wg2 = wg.reshape(m_per_g, c_g * kh * kw)
        out[:, g * m_per_g : (g + 1) * m_per_g] = (
            np.matmul(wg2, cols).reshape(n, m_per_g, ho, wo)
        )

    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return [out.astype(np.float32)]


@register_op("MatMul")
def op_matmul(node, inputs):
    a, b = inputs[0], inputs[1]
    return [np.matmul(a, b).astype(np.float32)]


@register_op("Gemm")
def op_gemm(node, inputs):
    """Y = alpha * A' @ B' + beta * C。"""
    a, b = inputs[0], inputs[1]
    alpha = node.attrs.get("alpha", 1.0)
    beta = node.attrs.get("beta", 1.0)
    trans_a = node.attrs.get("transA", 0)
    trans_b = node.attrs.get("transB", 0)
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    y = alpha * np.matmul(a, b)
    if beta != 0.0 and len(inputs) > 2:
        y = y + beta * inputs[2]
    return [y.astype(np.float32)]


@register_op("Relu")
def op_relu(node, inputs):
    return [np.maximum(inputs[0], 0).astype(np.float32)]


@register_op("Add")
def op_add(node, inputs):
    return [(inputs[0] + inputs[1]).astype(np.float32)]


@register_op("Mul")
def op_mul(node, inputs):
    return [(inputs[0] * inputs[1]).astype(np.float32)]


@register_op("Flatten")
def op_flatten(node, inputs):
    axis = node.attrs.get("axis", 1)
    x = inputs[0]
    lead = int(np.prod(x.shape[:axis])) if axis > 0 else 1
    return [x.reshape(lead, -1).astype(np.float32)]


@register_op("Reshape")
def op_reshape(node, inputs):
    """ONNX Reshape: data + shape（第二个输入），0 表示复制原维度，-1 表示自动推断。"""
    data, shape = inputs[0], np.asarray(inputs[1]).astype(np.int64)
    resolved = list(data.shape)
    for i, s in enumerate(shape):
        if s == 0:
            shape[i] = resolved[i]
    neg = [i for i, s in enumerate(shape) if s == -1]
    if neg:
        known = int(np.prod([s for s in shape if s != -1]))
        shape[neg[0]] = int(np.prod(data.shape)) // known
    return [data.reshape(tuple(shape)).astype(np.float32)]


@register_op("MaxPool")
def op_maxpool(node, inputs):
    x = inputs[0]
    kernel = node.attrs.get("kernel_shape", [2, 2])
    strides = node.attrs.get("strides", kernel)
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    kh, kw = kernel
    sh, sw = strides
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    n, c, h, w = x.shape
    ho = (xp.shape[2] - kh) // sh + 1
    wo = (xp.shape[3] - kw) // sw + 1
    out = np.full((n, c, ho, wo), -np.inf, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out = np.maximum(out, xp[:, :, i : i + ho * sh : sh, j : j + wo * sw : sw])
    return [out.astype(np.float32)]


@register_op("Sigmoid")
def op_sigmoid(node, inputs):
    x = inputs[0]
    return [(1.0 / (1.0 + np.exp(-x))).astype(np.float32)]


@register_op("Softmax")
def op_softmax(node, inputs):
    axis = node.attrs.get("axis", 1)
    x = inputs[0]
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return [(e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)]
