"""控制台启动入口：监听平台注入的 PORT（默认 8000），绑定 0.0.0.0。

本地:  python3 webapp/run.py
云端:  由发布平台注入 PORT 环境变量后启动
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("webapp.server:app", host="0.0.0.0", port=port, log_level="info")
