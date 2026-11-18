"""边缘 AI 运行时 · 可视化控制台（FastAPI 后端）。

启动:
    cd edge-ai-runtime
    python3 -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000

浏览器打开 http://localhost:8000
"""

import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import EdgeRuntime, list_backends          # noqa: E402
from runtime.graph.loader import load_onnx             # noqa: E402

MODEL_PATH = ROOT / "examples" / "test_model.onnx"

app = FastAPI(title="EdgeAI Runtime Console")
_sessions: dict[str, EdgeRuntime] = {}


def get_session(backend: str) -> EdgeRuntime:
    """按后端名缓存会话（真实 NPU 场景即 context 常驻）。"""
    if backend not in _sessions:
        rt = EdgeRuntime(backend=backend, verbose=False)
        rt.load(str(MODEL_PATH))
        b = rt.backend()
        if hasattr(b, "_verbose"):
            b._verbose = False
        _sessions[backend] = rt
    return _sessions[backend]


class InferReq(BaseModel):
    backend: str = "ort_cpu"
    seed: int = 0


@app.get("/api/backends")
def api_backends():
    return {
        "model": MODEL_PATH.name,
        "model_exists": MODEL_PATH.exists(),
        "backends": list_backends(),
    }


@app.get("/api/graph")
def api_graph():
    _, ir = load_onnx(MODEL_PATH)
    return {
        "name": ir.name,
        "opset": ir.opset,
        "inputs": ir.inputs,
        "outputs": ir.outputs,
        "nodes": [
            {
                "name": n.name,
                "op": n.op_type,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "attrs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in n.attrs.items()},
            }
            for n in ir.nodes
        ],
        "weights": len(ir.initializers),
        "weight_kb": round(sum(w.nbytes for w in ir.initializers.values()) / 1024, 2),
    }


def _feed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"input": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)}


@app.post("/api/infer")
def api_infer(req: InferReq):
    rt = get_session(req.backend)
    feed = _feed(req.seed)

    t0 = time.perf_counter()
    out = rt.run(**feed)
    ms = (time.perf_counter() - t0) * 1000

    arr = out["output"]
    return {
        "backend": rt.backend().name,
        "latency_ms": round(ms, 3),
        "shape": list(arr.shape),
        "values": [round(float(v), 4) for v in arr.flatten()[:10]],
        "argmax": int(np.argmax(arr)),
    }


@app.post("/api/compare")
def api_compare(req: InferReq):
    """双后端一致性对比：同一输入分别推理，比较结果与耗时。"""
    feed = _feed(req.seed)
    results, timings = {}, {}
    for name in ("ort_cpu", "sim_npu"):
        try:
            rt = get_session(name)
            t0 = time.perf_counter()
            out = rt.run(**feed)
            timings[name] = round((time.perf_counter() - t0) * 1000, 3)
            results[name] = out["output"]
        except Exception as e:  # 后端不可用时降级展示
            return {"error": f"{name}: {e}"}

    a, b = results["ort_cpu"], results["sim_npu"]
    return {
        "seed": req.seed,
        "timings": timings,
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "consistent": bool(np.allclose(a, b, atol=1e-4)),
        "argmax": {"ort_cpu": int(np.argmax(a)), "sim_npu": int(np.argmax(b))},
        "values": {
            "ort_cpu": [round(float(v), 4) for v in a.flatten()[:10]],
            "sim_npu": [round(float(v), 4) for v in b.flatten()[:10]],
        },
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
