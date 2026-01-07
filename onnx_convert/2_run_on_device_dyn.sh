#!/bin/bash

echo '--- Running Device Script ---'
export QNN_SDK_ROOT='/usr'

echo '--- Checking for Python Inference ---'
if command -v python3 &> /dev/null; then
   echo 'Python3 found.'
   echo 'Creating virtual environment...'
   python3 -m venv venv
   source venv/bin/activate
   echo 'Installing Python dependencies (onnxruntime, numpy)...'
   pip install onnxruntime numpy
   echo 'Running: python inference_qnn_onnx.py --model dinov3.onnx --backend qnn_cpu --qnn_lib /usr/lib/libQnnCpu.so'
   python inference_qnn_onnx.py --model dinov3.onnx --backend qnn_cpu --qnn_lib /usr/lib/libQnnCpu.so
else
   echo 'Python3 not found.'
fi
