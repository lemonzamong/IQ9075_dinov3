
import os
import argparse
from huggingface_hub import snapshot_download, login

def download_dinov3_pth(model_id, token=None, output_dir=None, allow_patterns=None, force_download=False):
    """
    Downloads the DINOv3 model (PyTorch weights) from Hugging Face.
    """
    if token:
        print("Logging in to Hugging Face...")
        login(token=token)
    
    print(f"Downloading model: {model_id}...")
    if allow_patterns:
        print(f"Downloading only files matching: {allow_patterns}")
    
    # If output_dir is not specified, it downloads to the default HF cache.
    # If specified, it downloads/copies to that directory.
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir=output_dir,
            local_dir_use_symlinks=False, # Set to False to get actual files if using local_dir
            resume_download=True,
            allow_patterns=allow_patterns,
            force_download=force_download
        )
        print(f"Successfully downloaded to: {local_dir}")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DINOv3 PyTorch Model")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vit7b16-pretrain-lvd1689m", help="Target Model ID")
    parser.add_argument("--token", type=str, default="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", help="Hugging Face Auth Token")
    parser.add_argument("--output_dir", type=str, default="dinov3_vit7b_pth", help="Directory to save the model")
    parser.add_argument("--allow_patterns", nargs='+', help="Patterns of files to download (e.g., *.safetensors)")
    parser.add_argument("--force_download", action="store_true", help="Force download even if exists")
    
    args = parser.parse_args()
    
    # Use the token provided in previous context if not passed explicitly? 
    # Better to force user to pass it or rely on cache.
    # I'll default to the one user gave earlier if I was running it, but for code generation I leave it optional.
    
    download_dinov3_pth(args.model_id, args.token, args.output_dir, args.allow_patterns, args.force_download)
