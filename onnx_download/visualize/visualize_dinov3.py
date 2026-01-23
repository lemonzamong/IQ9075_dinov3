import argparse
import onnxruntime as ort
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path, size=224):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1) # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0) # Add batch dimension
    return img_data, img, original_size

def visualize_tokens(model_path, image_path, output_path, patch_size=14): # Default patch size usually 14 for DINOv2 but let's check
    # Note: DINOv3-vitb16 likely has patch size 16. The name says 'vitb16'.
    if 'vitb16' in model_path:
        patch_size = 16
    elif 'vitl14' in model_path:
        patch_size = 14
    
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get input size from model if possible, else default 224
    input_shape = session.get_inputs()[0].shape
    h, w = 224, 224
    if len(input_shape) == 4:
        if isinstance(input_shape[2], int): h = input_shape[2]
        if isinstance(input_shape[3], int): w = input_shape[3]
    
    print(f"Using input size: {h}x{w} (Patch size assumption: {patch_size})")

    input_data, pil_img, original_size = preprocess_image(image_path, size=h)
    
    input_name = session.get_inputs()[0].name
    # Run inference
    # We expect 'last_hidden_state' or similar. 
    # Let's check outputs
    output_names = [o.name for o in session.get_outputs()]
    print(f"Model outputs: {output_names}")
    
    outputs = session.run(None, {input_name: input_data})
    
    # Assuming the first output is the sequence of tokens
    last_hidden_state = outputs[0] 
    print(f"Output shape: {last_hidden_state.shape}")
    
    # Shape: (Batch, Sequence, Dim)
    # Sequence = 1 (CLS) + n_patches + registers (maybe)
    # For 224x224 and patch 16: 14x14 = 196 patches.
    # Total tokens = 197 usually.
    
    tokens = last_hidden_state[0] # remove batch dim -> (Seq, Dim)
    
    # Heuristic to find patch tokens
    n_patches = (h // patch_size) * (w // patch_size)
    print(f"Expected patches: {n_patches}")
    
    if tokens.shape[0] == n_patches:
        patch_tokens = tokens
    elif tokens.shape[0] > n_patches:
        # Assuming CLS token is first and registers are after or before. 
        # Usually CLS is at 0.
        # DINOv2 registers are usually at the end? Or after CLS?
        # Let's assume standard ViT: CLS at 0, then patches. 
        # But if registers exist, they might be after CLS.
        # Let's take the last n_patches
        patch_tokens = tokens[-n_patches:]
    else:
        print(f"Error: Token count {tokens.shape[0]} is less than expected patches {n_patches}")
        return

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=3)
    pca_tokens = pca.fit_transform(patch_tokens)
    
    # Normalize to 0-1
    pca_tokens = (pca_tokens - pca_tokens.min(0)) / (pca_tokens.max(0) - pca_tokens.min(0))
    
    # Reshape to grid
    grid_h = h // patch_size
    grid_w = w // patch_size
    pca_img = pca_tokens.reshape(grid_h, grid_w, 3)
    
    # Resize to original image size for overlay/comparison
    pca_img_pil = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img_resized = pca_img_pil.resize(original_size, Image.Resampling.NEAREST)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(Image.open(image_path))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(pca_img_resized)
    ax[1].set_title(f"PCA Visualization (Patch {patch_size})")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    visualize_tokens(args.model_path, args.image_path, args.output_path)
