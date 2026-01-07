#!/bin/bash

# 0. Host Setup
# Run this script on your x86_64 Host Machine (Ubuntu 20.04/22.04)
# Ensure QNN SDK is installed and QNN_SDK_ROOT is set.

# Check for QNN_SDK_ROOT
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT is not set."
    echo "Please set it: export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/<version>"
    exit 1
fi

# Source QNN Environment (adjust path if needed)
# Standard location: $QNN_SDK_ROOT/bin/envsetup.sh
if [ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ]; then
    source "$QNN_SDK_ROOT/bin/envsetup.sh"
else
    echo "Warning: envsetup.sh not found. Assuming environment is already set up or manual path config."
fi

MODEL_NAME="dinov3"
ONNX_FILE="${MODEL_NAME}.onnx"
OUTPUT_CPP="${MODEL_NAME}_qnn.cpp"
OUTPUT_BIN="${MODEL_NAME}_qnn.bin"
OUTPUT_LIB_DIR="./libs"

# 1. Check if ONNX model exists
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: $ONNX_FILE not found. Please run export_dinov3.py first."
    exit 1
fi

echo "--- Step 1: Converting ONNX to QNN Graph (C++/Binary) ---"
# Note: input_list.txt is optional for basic conversion but recommended for quantization.
# We skip it for pure float conversion unless required by specific SDK version.
qnn-onnx-converter \
    --input_network "$ONNX_FILE" \
    --output_path "$OUTPUT_CPP" \
    --input_list input_list.txt \
    --dry_run \
    --no_simplification # Sometimes helps with complex models like ViT

if [ $? -ne 0 ]; then
    echo "Error: qnn-onnx-converter failed."
    exit 1
fi

echo "--- Step 2: Generating Model Library for Aarch64 ---"
echo "Targeting: x86_64-linux-clang (Host) and aarch64-linux-clang (Device)"

# We generate for Aarch64 (Device)
# Note: We need aarch64 compiler available (aarch64-linux-gnu-g++ or clang with target)
# The QNN SDK usually provides a clang toolchain or uses system cross-compiler.
# Assuming standard QNN setup matches 'aarch64-linux-clang' config.

mkdir -p $OUTPUT_LIB_DIR

qnn-model-lib-generator \
    -c "$OUTPUT_CPP" \
    -b "$OUTPUT_BIN" \
    -o "$OUTPUT_LIB_DIR" \
    -t aarch64-linux-clang 

if [ $? -ne 0 ]; then
    echo "Error: qnn-model-lib-generator failed."
    echo "Ensure you have the QNN SDK correctly configured for aarch64 targets."
    exit 1
fi

echo "Success! Library generated at $OUTPUT_LIB_DIR/aarch64-linux-clang/lib${MODEL_NAME}.so"
echo "Next: Transfer 'libs/', 'inference_dinov3.cpp', 'CMakeLists.txt', and 'build_and_run_device.sh' to your IQ-9075 device."
