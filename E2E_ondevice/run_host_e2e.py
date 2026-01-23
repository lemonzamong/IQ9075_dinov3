
import argparse
import os
import subprocess
import paramiko
from scp import SCPClient
import shutil
import sys
import time

# Import our custom export logic
# Legacy local import removed as we only use this script for transfer
# try:
#     from export_model import export_model_logic
# except ImportError:
#     print("Error: export_model.py not found in current directory.")
#     sys.exit(1)

# Device Config
DEVICE_IP = "192.168.0.202"
USERNAME = "ubuntu"
PASSWORD = "qualcomm"
REMOTE_BASE_DIR = "/home/ubuntu/dinov3_e2e"

def create_ssh_client(server, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=user, password=password)
    return client

def run_remote_command(ssh, command, stream=True):
    print(f"[REMOTE] {command}")
    # get_pty=True merges stdout/stderr and allows line-buffering usually
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    
    full_output = ""
    if stream:
        # Stream line by line
        for line in iter(stdout.readline, ""):
            print(line, end="")
            full_output += line
            
    exit_status = stdout.channel.recv_exit_status()
    # If not streaming, we still need to read remaining if any
    if not stream:
        full_output = stdout.read().decode()
        
    return exit_status, full_output, "" # Stderr merged in stdout due to Pty

# Progress Callback
def progress(filename, size, sent):
    sys.stdout.write(f"\rUploading {filename}: {float(sent)/float(size)*100:.2f}%")
    sys.stdout.flush()
    if sent == size:
        sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Host-Side E2E Workflow (Host Export -> Device Inference)")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Hugging Face Model ID")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face Auth Token")
    parser.add_argument("--skip_export", action="store_true", help="Skip export/conversion if artifacts exist")
    args = parser.parse_args()
    
    model_id = args.model_id
    safe_name = model_id.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("-", "_")
    if "dinov3" not in safe_name: safe_name = "dinov3_" + safe_name
    
    print(f"--- Host-Side Workflow for {model_id} ---")
    
    # 1. Local Export
    onnx_file = f"assets/{safe_name}.onnx"
    # Ensure assets dir exists
    if not os.path.exists("assets"):
        os.makedirs("assets")

    if not args.skip_export:
        print("\n--- Step 1: Local Export (PyTorch -> ONNX) ---")
        # We invoke export_model.py logic directly via subprocess or import
        # Subprocess is safer for env isolation if needed, but we checked deps.
        # We'll use subprocess to stream output clearly.
        cmd = [sys.executable, "export_model.py", "--model_id", model_id, "--output_file", onnx_file]
        if args.auth_token:
            cmd.extend(["--auth_token", args.auth_token])
            
        ret = subprocess.call(cmd)
        if ret != 0:
            print("Local Export failed.")
            sys.exit(1)
            
        # 2. Local Conversion
        print("\n--- Step 2: Local Conversion (ONNX -> QNN C++) ---")
        # SDK Path hardcoded based on discovery
        SDK_ROOT = "/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/v2.41.0.251128/qairt/2.41.0.251128"
        converter = f"{SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter"
        
        cpp_file = f"assets/{safe_name}_qnn.cpp"
        bin_file = f"assets/{safe_name}_qnn.bin"
        
        conv_cmd = [
            converter,
            "--input_network", onnx_file,
            "--output_path", cpp_file,
            "--input_dim", "pixel_values", "1,3,224,224",
            "--generate_types", "Cpu"
        ]
        
        print(f"Running: {' '.join(conv_cmd)}")
        ret = subprocess.call(conv_cmd)
        if ret != 0:
            print("Local Conversion failed.")
            sys.exit(1)
            
    else:
        print("Skipping Export/Conversion as requested.")
        cpp_file = f"assets/{safe_name}_qnn.cpp"
        bin_file = f"assets/{safe_name}_qnn.bin"

    # 3. Upload Artifacts
    print("\n--- Step 3: Uploading Artifacts to Device ---")
    ssh = create_ssh_client(DEVICE_IP, USERNAME, PASSWORD)
    scp = SCPClient(ssh.get_transport(), socket_timeout=3600.0, progress=progress)
    
    # Ensure remote dir
    run_remote_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}", stream=False)
    
    # Files to upload: cpp, bin, device_inference.py, inference_dinov3.cpp (if needed)
    # Actually device_inference.py builds the cpp.
    # verify existence
    if not os.path.exists(cpp_file):
        print(f"Error: {cpp_file} missing.")
        sys.exit(1)
        
    uploads = [cpp_file, bin_file, "device_inference.py", "../onnx_convert/native_qnn/src/inference_dinov3.cpp"]
    
    for f in uploads:
        if os.path.exists(f):
            size_bytes = os.path.getsize(f)
            print(f"Uploading {os.path.basename(f)} ({size_bytes/1024/1024:.2f} MB)...")
            scp.put(f, remote_path=f"{REMOTE_BASE_DIR}/{os.path.basename(f)}")
        else:
    # Extract Assets on Device
    print("Extracting assets on device...")
    cmds = [
        f"tar -xzf {REMOTE_BASE_DIR}/sdk_headers.tar.gz -C {REMOTE_BASE_DIR} 2>/dev/null",
        f"mkdir -p {REMOTE_BASE_DIR}/lib && tar -xzf {REMOTE_BASE_DIR}/sdk_libs.tar.gz -C {REMOTE_BASE_DIR}/lib 2>/dev/null",
        f"mkdir -p {REMOTE_BASE_DIR}/lib/hexagon && tar -xzf {REMOTE_BASE_DIR}/skel_libs.tar.gz -C {REMOTE_BASE_DIR}/lib 2>/dev/null"
    ]
    for c in cmds:
        run_remote_command(ssh, c, stream=False)

    # 2. Trigger Device Orchestrator
    print("\nStep 2: Triggering Device Orchestrator...")
    # Pass token if needed? For now assuming public or simple
    # We need to make sure python3 has requirements.
    # We can try to install them? No, user said "Qualcomm PC", assume Env is ready.
    
    cmd = f"cd {REMOTE_BASE_DIR} && python3 device_orchestrator.py --model_id {model_id}"
    if args.auth_token:
        cmd += f" --auth_token {args.auth_token}"
    exit_code, out, err = run_remote_command(ssh, cmd)
    
    print("\n--- Final Status ---")
    if exit_code == 0:
        print("Success!")
    else:
        print("Failure.")
        
    ssh.close()

if __name__ == "__main__":
    main()
