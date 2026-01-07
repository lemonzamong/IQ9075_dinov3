#!/bin/bash
set -e

# Assumes running from onnx_convert directory
echo "Creating directory structure..."
mkdir -p native_qnn/src native_qnn/bin
mkdir -p assets
mkdir -p scripts
mkdir -p test

echo "Moving Source Files..."
[ -f "native_qnn/dinov3_qnn.cpp" ] && mv native_qnn/dinov3_qnn.cpp native_qnn/src/
[ -f "native_qnn/inference_dinov3.cpp" ] && mv native_qnn/inference_dinov3.cpp native_qnn/src/
[ -f "native_qnn/dinov3_qnn.bin" ] && mv native_qnn/dinov3_qnn.bin native_qnn/bin/

echo "Moving Assets..."
[ -f "native_qnn/dinov3_qnn_net.json" ] && mv native_qnn/dinov3_qnn_net.json assets/
[ -f "dinov3.onnx" ] && mv dinov3.onnx assets/
[ -f "dinov3.onnx.data" ] && mv dinov3.onnx.data assets/

echo "Moving Scripts..."
[ -f "export_dinov3.py" ] && mv export_dinov3.py scripts/
[ -f "preprocess_input.py" ] && mv preprocess_input.py scripts/

echo "Moving Test Files..."
[ -f "test_image.jpg" ] && cp test_image.jpg test/ # Keep copy? No move.
[ -f "test_image.jpg" ] && mv test_image.jpg test/
[ -f "input.raw" ] && mv input.raw test/
[ -f "input_list.txt" ] && mv input_list.txt test/

echo "Cleaning up..."
rm -rf tmp_* libs_x86 build inputs input.raw input_list.txt sdk_libs.tar.gz sdk_headers.tar.gz sdk_jni.tar.gz run_on_device.sh output

echo "Restructuring Complete."
