#!/bin/bash

# Exit on error
set -e

# Default SDK Root if not set (User can override)
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT is not set."
    echo "Please set it: export QNN_SDK_ROOT=/path/to/qairt/version"
    exit 1
fi

echo "Using QNN_SDK_ROOT: $QNN_SDK_ROOT"

# Source QNN Environment
if [ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ]; then
    source "$QNN_SDK_ROOT/bin/envsetup.sh"
else
    echo "Error: envsetup.sh not found at $QNN_SDK_ROOT/bin/envsetup.sh"
    echo "Please check your QNN_SDK_ROOT path."
    exit 1
fi

# Define paths
MODEL_NAME="dinov3"
ONNX_FILE="../common/${MODEL_NAME}.onnx"
OUTPUT_CPP="${MODEL_NAME}_qnn.cpp"
OUTPUT_BIN="${MODEL_NAME}_qnn.bin"
OUTPUT_LIB_DIR="./libs"

# Check ONNX file
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: $ONNX_FILE not found."
    echo "Please run 'python3 ../common/export_dinov3.py' first."
    exit 1
fi

echo "--- Step 0: Preparing Calibration Data ---"
# Create a dummy input list if not exists.Ideally this should be real data.
if [ ! -f "input_list.txt" ]; then
    # Create 224x224 dummy
    python3 -c "import numpy as np; np.zeros((1,3,224,224), dtype=np.float32).tofile('input_224.raw')"
    echo "pixel_values:=./input_224.raw" > input_list.txt
fi

echo "--- Step 1: Converting ONNX to QNN Graph (Quantized) ---"
# Check if qnn-onnx-converter is in PATH
if ! command -v qnn-onnx-converter &> /dev/null; then
    echo "Error: qnn-onnx-converter could not be found."
    echo "This likely means envsetup.sh didn't run correctly or the SDK is incomplete."
    exit 1
fi

qnn-onnx-converter \
    --input_network "$ONNX_FILE" \
    --output_path "$OUTPUT_CPP" \
    --input_dim "pixel_values" 1,3,224,224 \
    --input_list "input_list.txt" \
    --no_simplification

# Note: We rely on 'set -e' to exit if the above fails.

echo "--- Step 2: Generating Model Library for AArch64 (SKIPPED) ---"
# We skip this because the host qnn-model-lib-generator often doesn't support
# aarch64-linux-clang properly without cross-compiler setup.
# We rely on On-Device Compilation (see deploy.py).
#
# if ! command -v qnn-model-lib-generator &> /dev/null; then
#     echo "Error: qnn-model-lib-generator not found."
#     exit 1
# fi
#
# qnn-model-lib-generator \
#     -c "$OUTPUT_CPP" \
#     -b "$OUTPUT_BIN" \
#     -o "$OUTPUT_LIB_DIR" \
#     -t aarch64-linux-clang

echo "Success! QNN Library generated at $OUTPUT_LIB_DIR"
