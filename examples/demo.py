"""演示：同一份代码，切换后端运行同一个 ONNX 模型。

运行方式:
    cd edge-ai-runtime
    python examples/demo.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# 保证从任意工作目录运行都能找到 runtime 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime import EdgeRuntime, list_backends  # noqa: E402

MODEL = Path(__file__).resolve().parent / "test_model.onnx"


def make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {"input": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)}


def run_on(backend: str, feed):
    print(f"\n{'=' * 60}\n后端: {backend}\n{'=' * 60}")
    rt = EdgeRuntime(backend=backend)
    rt.load(str(MODEL))

    t0 = time.perf_counter()
    outputs = rt.run(**feed)
    elapsed = (time.perf_counter() - t0) * 1000
    out = outputs["output"]
    print(f"[EdgeRuntime] 推理耗时 {elapsed:.2f}ms, 输出形状 {out.shape}")
    print(f"[EdgeRuntime] 输出(top3): {out.flatten()[:3]}")
    return out


def main():
    print("已注册后端:")
    for name, meta in list_backends().items():
        flag = "可用" if meta["available"] else "不可用(缺依赖)"
        print(f"  - {name}: {meta['description']} [{flag}]")

    feed = make_inputs()

    # 同一份调用代码，只换后端参数
    out_ort = run_on("ort_cpu", feed)
    out_npu = run_on("sim_npu", feed)

    print(f"\n{'=' * 60}\n一致性校验: ort_cpu vs sim_npu\n{'=' * 60}")
    max_diff = np.max(np.abs(out_ort - out_npu))
    ok = np.allclose(out_ort, out_npu, atol=1e-4)
    print(f"最大绝对误差: {max_diff:.2e}  ->  {'通过' if ok else '未通过'}")

    # 展示 auto 模式
    print(f"\n{'=' * 60}\nauto 模式（自动选择可用后端）\n{'=' * 60}")
    rt = EdgeRuntime(backend="auto", verbose=False)
    rt.load(str(MODEL))
    print(f"[auto] 实际选择后端: {rt.backend_name} / {rt.backend().name}")
    outputs = rt.run(**feed)
    print(f"[auto] 推理成功, 输出形状: {outputs['output'].shape}")


if __name__ == "__main__":
    main()
