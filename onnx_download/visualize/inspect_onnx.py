import onnxruntime as ort
import numpy as np
import sys

def inspect_model(model_path):
    print(f"Inspecting {model_path}...")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Inputs:")
    for i in session.get_inputs():
        print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    print("Outputs:")
    for o in session.get_outputs():
        print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
    print("-" * 20)

model_paths = [
    "/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/onnx_download/dinov3-vitb16/dinov3.onnx",
    "/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/onnx_download/dinov3-vitb7b16/dinov3.onnx"
]

for p in model_paths:
    inspect_model(p)
