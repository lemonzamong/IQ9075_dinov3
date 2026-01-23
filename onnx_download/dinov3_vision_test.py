import torch
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

from dinov3.models.vision_transformer import vit_base

def compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord):
    target_idx = target_patch_coord[0] * W + target_patch_coord[1]
    target_feature = patch_features[0, target_idx]
    similarities = F.cosine_similarity(
        target_feature.unsqueeze(0),
        patch_features[0],
        dim=1
    )
    heatmap = similarities.reshape(H, W).cpu().numpy()
    return heatmap

def plot_similarity_heatmap(heatmap, target_patch_coord, save_path=None):
    H, W = heatmap.shape
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap='viridis', aspect='equal')
    ax.plot(target_patch_coord[1], target_patch_coord[0], 'ro', markersize=10)
    plt.colorbar(im, ax=ax, label='Cosine Similarity') 
    ax.set_xlabel('Width (patch index)')
    ax.set_ylabel('Height (patch index)')
    ax.set_title(f'Cosine Similarity to Patch at {target_patch_coord}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return fig, ax

def main(image_path, model_path, patch_size=16, input_size=224, output_path="patch_similarity_heatmap.png"):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    tensor_image = transform(image).unsqueeze(0)  # (1, 3, H, W)

    model = vit_base(patch_size=patch_size)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        tensor_image = tensor_image.cuda()

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        features_dict = model.forward_features(tensor_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        patch_features = features_dict["x_norm_patchtokens"]
        num_patches = patch_features.shape[1]
        H = W = int(num_patches ** 0.5)
        target_patch_coord = (H // 2, W // 2)
        heatmap = compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord)
        plot_similarity_heatmap(heatmap, target_patch_coord, save_path=output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to DINOv3 .pth model')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size (default=16)')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size (default=224)')
    parser.add_argument('--output', type=str, default='patch_similarity_heatmap.png', help='Output heatmap image path')
    args = parser.parse_args()
    main(args.image, args.model, args.patch_size, args.input_size, args.output)