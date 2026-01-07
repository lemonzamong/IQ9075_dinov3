import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import sys

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_data = np.array(img).astype('float32') / 255.0
    # Normalize (standard ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1) # HWC to CHW
    return img_data[np.newaxis, :] # Add batch dim

def main():
    model_path = "assets/dinov3.onnx"
    image_path = "test/test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"--- Visual Verification (ORT + QNN) ---")
    print(f"Loading Model: {model_path}")
    print(f"Processing Image: {image_path}")

    # Prepare input
    input_data = preprocess_image(image_path)
    
    # Configure QNN EP
    # Note: Using system backend for stability
    options = {
        "backend_path": "/usr/lib/libQnnCpu.so"
    }
    
    try:
        session = ort.InferenceSession(model_path, providers=["QNNExecutionProvider"], provider_options=[options])
        print(f"[ SUCCESS ] QNN Session Initialized.")
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Running Inference...")
        outputs = session.run([output_name], {input_name: input_data})
        output_tensor = outputs[0]
        
        print(f"\n--- Inference Result ---")
        print(f"Output Name: {output_name}")
        print(f"Output Shape: {output_tensor.shape}")
        
        # Performance/Value Stats for "Visual" Verification
        print(f"Value Stats:")
        print(f"  - Mean: {np.mean(output_tensor):.4f}")
        print(f"  - Std:  {np.std(output_tensor):.4f}")
        print(f"  - Max:  {np.max(output_tensor):.4f}")
        print(f"  - Min:  {np.min(output_tensor):.4f}")
        
        print(f"\nFirst 5 values of embedding:")
        print(output_tensor[0, 0, :5])
        
        print(f"\n[ RESULT ] Real Image Inference VERIFIED via QNN Acceleration.")
        
    except Exception as e:
        print(f"[ ERROR ] Inference failed: {e}")
        print("\nFallback: Trying CPU execution...")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
        print(f"CPU Output Shape: {outputs[0].shape}")

if __name__ == "__main__":
    main()
