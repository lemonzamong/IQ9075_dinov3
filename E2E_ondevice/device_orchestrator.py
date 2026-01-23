
import os
import argparse
import subprocess
import sys
import glob

# THIS SCRIPT RUNS ON THE IQ-9075 DEVICE

def run_command(command, stream_output=True):
    print(f"[DEVICE] {command}")
    
    if stream_output:
        # Direct execution to allow PTY/stdout/stderr to flow naturally
        # This prevents buffering issues and missing tracebacks
        ret = subprocess.call(command, shell=True)
        return ret, "", ""
    else:
        # Capture for internal logic (like checking pip list)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = process.communicate()
        return process.returncode, out, err

def main():
    parser = argparse.ArgumentParser(description="Device-side Orchestrator")
    parser.add_argument("--model_id", required=True, help="Hugging Face Model ID")
    parser.add_argument("--auth_token", default="", help="HF Token")
    args = parser.parse_args()
    
    model_id = args.model_id
    safe_name = model_id.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("-", "_")
    if "dinov3" not in safe_name: safe_name = "dinov3_" + safe_name
    
    base_dir = os.getcwd()
    print(f"--- Running E2E on Device for {model_id} ---")
    
    print("\n--- Step 0: Checking/Installing Dependencies ---")
    pkgs = ["torch", "transformers", "onnx", "huggingface_hub", "accelerate", "onnxscript"]
    to_install = []
    
    # Check what's missing
    for p in pkgs:
        check_p = p
        if p == "optimum": check_p = "optimum.exporters.onnx"
        
        ret, _, _ = run_command(f"python3 -c 'import {check_p}'")
        if ret != 0:
            to_install.append(p)
            
    if to_install:
        print(f"Missing packages: {to_install}. Installing... (This may take a while)")
        # Use --break-system-packages if running on newer Ubuntu/Debian externally managed env, 
        # but let's try standard pip first. If it fails due to PEP 668, we might need --break-system-packages user flag.
        # Safest is likely `pip3 install ... --break-system-packages` if on Ubuntu 24.04 (Noble), but 22.04 is fine.
        # User's uname said "Linux ubuntu 6.8.0...". This looks like a newer kernel/distro.
        # Let's add --break-system-packages just in case, or handle the error.
        # Actually, let's just try running it.
        
        # Install missing packages
        install_cmd = f"pip3 install {' '.join(to_install)} --break-system-packages"
        ret, _, _ = run_command(install_cmd)
        if ret != 0:
            print("Install failed. Retrying without --break-system-packages...")
            install_cmd = f"pip3 install {' '.join(to_install)}"
            ret, _, _ = run_command(install_cmd)
            if ret != 0:
                print("Failed to install dependencies.")
                sys.exit(1)
    else:
        print("All dependencies present.")

    print("\n--- Step 1: Exporting Model (PyTorch -> ONNX) ---")
    
    onnx_file = f"assets/{safe_name}.onnx"
    os.makedirs("assets", exist_ok=True)
    
    cmd = f"python3 -u export_model.py --model_id {model_id} --output_file {onnx_file}"
    if args.auth_token:
        cmd += f" --auth_token {args.auth_token}"
    
    ret, _, _ = run_command(cmd)
    if ret != 0:
        print(f"Export failed with return code {ret}.")
        sys.exit(1)
        
    # 2. Conversion (ONNX -> QNN C++)
    print("\n--- Step 2: Converting to QNN (ONNX -> Cpp) ---")
    # Need to find qnn-onnx-converter
    # It might be in $QNN_SDK_ROOT/bin/...
    # On device, QNN_SDK_ROOT might not be set, or might be different.
    # Searching common paths.
    
    # We pushed SDK libs to ./lib/aarch64...
    # But converter is usually NOT in the Runtime package.
    # However, if user "Qualcomm PC" implies developer kit with full SDK:
    qnn_sdk = os.environ.get("QNN_SDK_ROOT", "/opt/qcom/qnn-sdk")
    
    # Try generic qnn-onnx-converter name in path
    converter_cmd = "qnn-onnx-converter"
    
    # Check if we can find it
    ret, out, _ = run_command(f"which {converter_cmd}", stream_output=False)
    if ret != 0:
        # Not in path. Search in known sdk locations if we can
        print("qnn-onnx-converter not found in PATH. Checking QNN_SDK_ROOT...")
        if os.path.exists(qnn_sdk):
             # Try aarch64-linux-clang
             c = f"{qnn_sdk}/bin/aarch64-linux-clang/qnn-onnx-converter"
             if os.path.exists(c):
                 converter_cmd = c
             else:
                 print(f"Converter not found at {c}")
                 # Try x86 just in case (emulation?) No.
        else:
            print("SDK Root not found.")
            
    # Assuming we found it or will fail trying
    cpp_file = f"{safe_name}_qnn.cpp"
    bin_file = f"{safe_name}_qnn.bin"
    
    # Using pixel_values as input
    cmd = f"{converter_cmd} --input_network {onnx_file} --output_path {cpp_file} --input_dim pixel_values 1,3,224,224 --generate_types Cpu"
    
    ret, _, _ = run_command(cmd)
    if ret != 0:
        print("Conversion failed. Is qnn-onnx-converter installed on this device?")
        sys.exit(1)
        
    # 3. Inference
    print("\n--- Step 3: Device Inference ---")
    cmd = f"python3 device_inference.py --model_name {safe_name}"
    ret, _, _ = run_command(cmd)
    
    if ret == 0:
        print("\nE2E Workflow Complete: SUCCESS")
    else:
        print("\nE2E Workflow Complete: FAILURE")

if __name__ == "__main__":
    main()
