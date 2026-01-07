
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, output_path):
    print(f"Processing {image_path}...")
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img_data = np.array(img).astype(np.float32) / 255.0
        
        # Normalize (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # Transpose to CHW (1, 3, 224, 224)
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        
        # Save as raw
        img_data.tofile(output_path)
        print(f"Saved raw input to {output_path} (Shape: {img_data.shape}, Dtype: {img_data.dtype})")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output raw file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        exit(1)
    preprocess_image(args.input, args.output)
