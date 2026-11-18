"""端到端测试：验证两个后端加载、推理与结果一致性。

运行方式:
    cd edge-ai-runtime && python tests/test_backends.py
    （或 pytest tests/test_backends.py）
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.make_model import build_model, MODEL_PATH  # noqa: E402
from runtime import EdgeRuntime  # noqa: E402


def _ensure_model():
    if not MODEL_PATH.exists():
        import onnx

        MODEL_PATH.write_bytes(build_model().SerializeToString())


def _feed(seed=0):
    rng = np.random.default_rng(seed)
    return {"input": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)}


def test_ort_cpu_runs():
    _ensure_model()
    rt = EdgeRuntime(backend="ort_cpu", verbose=False)
    rt.load(str(MODEL_PATH))
    out = rt.run(**_feed())
    assert out["output"].shape == (1, 10)
    print("PASS: ort_cpu 推理成功")


def test_sim_npu_runs():
    _ensure_model()
    rt = EdgeRuntime(backend="sim_npu", verbose=False)
    rt.load(str(MODEL_PATH))
    out = rt.run(**_feed())
    assert out["output"].shape == (1, 10)
    print("PASS: sim_npu 推理成功")


def test_backends_consistent():
    _ensure_model()
    feed = _feed()
    out_a = EdgeRuntime(backend="ort_cpu", verbose=False)
    out_a.load(str(MODEL_PATH))
    out_b = EdgeRuntime(backend="sim_npu", verbose=False)
    out_b.load(str(MODEL_PATH))
    a = out_a.run(**feed)["output"]
    b = out_b.run(**feed)["output"]
    assert np.allclose(a, b, atol=1e-4), f"两后端结果不一致, 最大误差 {np.max(np.abs(a - b))}"
    print(f"PASS: 两后端结果一致 (最大误差 {np.max(np.abs(a - b)):.2e})")


def test_missing_input_raises():
    _ensure_model()
    rt = EdgeRuntime(backend="sim_npu", verbose=False)
    rt.load(str(MODEL_PATH))
    try:
        rt.run()  # 不传输入
        raise AssertionError("应当抛出 KeyError")
    except KeyError:
        print("PASS: 缺少输入时正确报错")


if __name__ == "__main__":
    test_ort_cpu_runs()
    test_sim_npu_runs()
    test_backends_consistent()
    test_missing_input_raises()
    print("\n全部测试通过 ✔")
