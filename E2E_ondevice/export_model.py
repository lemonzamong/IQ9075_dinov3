import torch
from transformers import AutoModel
import os
from huggingface_hub import login, snapshot_download
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Export DINOv3 model to ONNX")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Hugging Face model ID")
    parser.add_argument("--output_file", type=str, default="dinov3.onnx", help="Output ONNX filename")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face authentication token")
    return parser.parse_args()

def export_model_logic(model_id, output_file, auth_token=None):
    # User provided token
    token = auth_token if auth_token else os.environ.get("HF_TOKEN", "")
    
    if token:
        print(f"Logging in to Hugging Face...")
        login(token=token)
    else:
        print("No token provided, attempting to use cached credentials.")

    print(f"Downloading model snapshot: {model_id}...")
    try:
        # Download first to isolate network issues
        local_dir = snapshot_download(repo_id=model_id, token=token, resume_download=True)
        print(f"Model downloaded to: {local_dir}")
        model_path = local_dir
    except Exception as e:
        print(f"Download failed: {e}")
        model_path = model_id

    print(f"Loading model from: {model_path}")
    
    try:
        # load in fp16
        model = AutoModel.from_pretrained(
            model_path, 
            device_map="cpu", 
            trust_remote_code=True, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    # Create dummy input in FP16
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float16)

    print(f"Exporting to ONNX: {output_file} (FP16)")
    
    # Export using standard torch.onnx.export
    # Removing save_as_external_data as it's deprecated/removed in recent PyTorch.
    # We hope native export handles it or we catch the error.
    try:
        torch.onnx.export(
            model, 
            (dummy_input,), 
            output_file, 
            input_names=['pixel_values'], 
            output_names=['last_hidden_state', 'pooler_output'], 
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size'},
                'pooler_output': {0: 'batch_size'}
            }
        )
    except Exception as e:
        print(f"Export failed: {e}")
        # If it failed due to size, we might need a workaround, but let's see the error first.
        # Fallback to export_params=False?
        raise e
        
    print("Export complete.")

if __name__ == "__main__":
    args = get_args()
    export_model_logic(args.model_id, args.output_file, args.auth_token)
