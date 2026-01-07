import onnxruntime as ort
import numpy as np
import os
import argparse
import sys

def run_inference(model_path, backend='cpu', qnn_lib=''):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Check for external data file
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        print(f"Found external data file: {data_path}")
    else:
        print(f"Warning: External data file {data_path} not found locally. Inference might fail.")

    # Prepare session options
    sess_options = ort.SessionOptions()
    
    # Configure QNN provider options
    qnn_options = {}
    if qnn_lib:
        qnn_options['backend_path'] = qnn_lib
    
    providers = []
    if backend == 'htp':
        # QNN HTP
        if 'backend_path' not in qnn_options:
             qnn_options['backend_path'] = 'libQnnHtp.so'
        providers = [('QNNExecutionProvider', qnn_options), 'CPUExecutionProvider']
    elif backend == 'qnn_cpu':
        # QNN CPU 
        if 'backend_path' not in qnn_options:
             qnn_options['backend_path'] = 'libQnnCpu.so'
        providers = [('QNNExecutionProvider', qnn_options), 'CPUExecutionProvider']
    elif backend == 'cpu':
        # Standard ONNX Runtime CPU (non-QNN)
        providers = ['CPUExecutionProvider']
    else:
        print(f"Unknown backend {backend}, defaulting to CPU")
        providers = ['CPUExecutionProvider']

    print(f"Creating session with providers: {providers}")
    
    session = None
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
    except Exception as e:
        print(f"Failed to create inference session: {e}")
        return

    if session is None:
        print("Error: Session creation failed silently.")
        return

    # Get input details
    try:
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Model Input: {input_name}, Shape: {input_shape}")
    except Exception as e:
        print(f"Error getting input details: {e}")
        return

    # Prepare dummy input
    batch_size = 1
    c, h, w = 3, 224, 224 # Standard ViT size
    
    # Check if input shape is defined in model
    if len(input_shape) == 4 and isinstance(input_shape[2], int):
        h, w = input_shape[2], input_shape[3]

    dummy_input = np.random.randn(batch_size, c, h, w).astype(np.float32)

    # Run inference
    print("Running inference...")
    try:
        outputs = session.run(None, {input_name: dummy_input})
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")
        print("Inference completed successfully.")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dino v3 inference using ONNX Runtime QNN EP")
    parser.add_argument("--model", type=str, default="dinov3.onnx", help="Path to ONNX model")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "qnn_cpu", "htp"], help="Target backend")
    parser.add_argument("--qnn_lib", type=str, default="", help="Path to QNN backend library (e.g. libQnnCpu.so)")
    args = parser.parse_args()

    run_inference(args.model, args.backend, args.qnn_lib)
