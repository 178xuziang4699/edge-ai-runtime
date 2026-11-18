"""ONNX -> QNN context binary 离线转换驱动。

在装有 QNN/QAIRT SDK 的 x86 主机上运行：
    python tools/convert_qnn.py model.onnx --soc-arch v73

注意：三个工具的具体参数名随 SDK 版本略有差异，
以 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang 下的 --help 为准。
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

DEFAULT_SDK_BIN = Path("/opt/qnn/sdk/bin/x86_64-linux-clang")


def convert(onnx_path: str, out_dir: str, sdk_bin: Path,
            target: str = "aarch64-ubuntu-gcc7.5", soc_arch: str = "v73") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(onnx_path).stem

    # Step1: ONNX -> QNN 图表示（.cpp + .bin）
    subprocess.run(
        [str(sdk_bin / "qnn-onnx-converter"),
         "--input_network", onnx_path,
         "--output_path", str(out / stem)],
        check=True,
    )

    # Step2: 编译为目标平台库
    subprocess.run(
        [str(sdk_bin / "qnn-model-lib-generator"),
         "-c", str(out / f"{stem}.cpp"),
         "-b", str(out / f"{stem}.bin"),
         "-t", target],
        check=True,
    )

    # Step3: 生成设备端可加载的 context binary（HTP 后端，架构与芯片绑定）
    ctx_bin = out / f"{stem}_{soc_arch}.serialized.bin"
    subprocess.run(
        [str(sdk_bin / "qnn-context-binary-generator"),
         "--backend", "libQnnHtp.so",
         "--binary_file", str(out / f"{stem}.bin"),
         "--output_file", str(ctx_bin),
         "--backend_options",
         '{"log_level":"error","htp_arch":%s}' % soc_arch[1:]],
        check=True,
    )
    return ctx_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_model")
    ap.add_argument("--sdk-bin", default=str(DEFAULT_SDK_BIN))
    ap.add_argument("--soc-arch", default="v73", help="HTP 架构代际: v68/v69/v73/v75/v79")
    args = ap.parse_args()

    # 缓存指纹：模型内容哈希 + 芯片代际（context binary 不可跨芯片复用）
    model_hash = hashlib.sha256(Path(args.onnx_model).read_bytes()).hexdigest()[:16]
    out_dir = f"artifacts/{model_hash}_{args.soc_arch}"
    ctx_bin = convert(args.onnx_model, out_dir, Path(args.sdk_bin), soc_arch=args.soc_arch)
    print(f"转换完成: {ctx_bin}")


if __name__ == "__main__":
    sys.exit(main())
