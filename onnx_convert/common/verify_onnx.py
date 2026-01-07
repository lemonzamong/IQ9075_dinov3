import onnxruntime as ort
import numpy as np
import os

def verify_onnx():
    model_path = "dinov3.onnx"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"Verifying ONNX model: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input name: {input_name}, Shape: {input_shape}")
    
    # Handle dynamic batch size if present (usually represented as string or -1)
    batch_size = 1
    h, w = 224, 224
    if len(input_shape) == 4:
         # simple heuristic, assume NCHW
         pass
    
    dummy_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    
    outputs = session.run(None, {input_name: dummy_input})
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    print("Verification execution successful.")

if __name__ == "__main__":
    verify_onnx()
