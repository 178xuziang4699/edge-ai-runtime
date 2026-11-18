#!/usr/bin/env bash
# ONNX -> MediaTek Neuron (.dla) 离线转换脚本（Genio/开发板路线）。
# 设备环境：apt install mediatek-libneuron mediatek-neuron-utils mediatek-libneuron-dev
#
# 注意：ncc 主吃 TFLite；ONNX 通常先转 TFLite。
# 天玑手机平台走另一条工具链（mtk_model_converter/quantizer/compiler 产出 .nb），见设计文档。

set -euo pipefail

MODEL_TFLITE="$1"
OUTPUT_DIR="${2:-artifacts}"

mkdir -p "$OUTPUT_DIR"

# Step1: Neuron Compiler: tflite -> dla（与目标芯片绑定）
ncc-tflite \
    --input "$MODEL_TFLITE" \
    --output "$OUTPUT_DIR/model.dla" \
    --quantize INT8          # 可选：转换期量化

# Step2: 设备端冒烟验证（不接运行时即可确认模型合法性）
neuronrt -v "$OUTPUT_DIR/model.dla"

echo "转换完成: $OUTPUT_DIR/model.dla"
