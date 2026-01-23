import torch
from transformers import AutoImageProcessor, AutoModel
import os
from huggingface_hub import login
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Export DINOv3 model to ONNX")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Hugging Face model ID")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face authentication token")
    return parser.parse_args()

def export_model():
    args = get_args()
    
    # User provided token
    token = args.auth_token if args.auth_token else os.environ.get("HF_TOKEN", "")
    
    if token:
        print(f"Logging in to Hugging Face...")
        login(token=token)
    else:
        print("No token provided, attempting to use cached credentials.")

    model_name = args.model_id
    print(f"Loading model: {model_name}")
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy input
    # Expected input shape for ViT-B/16 usually includes pixel_values
    # Check processor config or default size if needed, usually 224x224 or larger.
    # We will use a standard size.
    dummy_input = torch.randn(1, 3, 224, 224)
    
    output_file = "dinov3.onnx"
    print(f"Exporting to ONNX: {output_file}")

    # Export
    torch.onnx.export(
        model, 
        (dummy_input,), 
        output_file, 
        input_names=['pixel_values'], 
        output_names=['last_hidden_state', 'pooler_output'], # Typical outputs, check model forward
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'}
        }
    )
    print("Export complete.")

if __name__ == "__main__":
    export_model()
