
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
    parser = argparse.ArgumentParser(description="E2E DINOv3 Host-to-Device Workflow")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Hugging Face Model ID")
    parser.add_argument("--qnn_sdk_root", type=str, default="/opt/qcom/qnn-sdk", help="Path to QNN SDK on Host") 
    parser.add_argument("--skip_transfer", action="store_true", help="Skip the file transfer step")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face Auth Token")
    args = parser.parse_args()
    
    model_id = args.model_id
    # Derive safe name: "facebook/dinov3-vitb16..." -> "dinov3_vitb16"
    safe_name = model_id.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("-", "_")
    if "dinov3" not in safe_name: safe_name = "dinov3_" + safe_name
    
    print(f"--- Workflow for {model_id} (Alias: {safe_name}) ---")
    print("Mode: Fully On-Device (Download -> Export -> Convert -> Run)")
    
    # 1. Transfer Scripts & Environment
    print("Step 1: Transferring Scripts to Device...")
    ssh = create_ssh_client(DEVICE_IP, USERNAME, PASSWORD)
    # Increase socket timeout for large files (99MB+)
    scp = SCPClient(ssh.get_transport(), socket_timeout=3600.0, progress=progress)
    
    # Clean/Create Dir
    run_remote_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}", stream=False)
    
    # Upload Scripts
    files_to_upload = [
        "export_model.py",
        "device_inference.py",
        "device_orchestrator.py",
        "../onnx_convert/native_qnn/src/inference_dinov3.cpp" # Harness
    ]
    
    # Check args.qnn_sdk_root
    qnn_sdk = args.qnn_sdk_root
    if not os.path.exists(qnn_sdk):
         # Fallback search
        possible = [
            os.environ.get("QNN_SDK_ROOT"),
            "/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/v2.41.0.251128/qairt/2.41.0.251128"
        ]
        for p in possible:
            if p and os.path.exists(p):
                qnn_sdk = p
                break
    
    if qnn_sdk and os.path.exists(qnn_sdk) and not args.skip_transfer:
        print(f"Found Host SDK at {qnn_sdk}. Syncing headers/libs to ensure device has them...")
        # Headers
        if not os.path.exists("sdk_headers.tar.gz"): subprocess.run(f"tar -czf sdk_headers.tar.gz -C \"{qnn_sdk}\" include", shell=True)
        files_to_upload.append("sdk_headers.tar.gz")
        # Libs
        if not os.path.exists("sdk_libs.tar.gz"): subprocess.run(f"tar -czf sdk_libs.tar.gz -C \"{qnn_sdk}/lib/aarch64-oe-linux-gcc11.2\" .", shell=True)
        files_to_upload.append("sdk_libs.tar.gz")
        # Hexagon
        if not os.path.exists("skel_libs.tar.gz"): subprocess.run(f"tar -czf skel_libs.tar.gz -C \"{qnn_sdk}/lib\" hexagon-v68 hexagon-v69 hexagon-v73 hexagon-v75", shell=True, stderr=subprocess.DEVNULL)
        files_to_upload.append("skel_libs.tar.gz")
    elif args.skip_transfer:
        print("Skipping SDK Asset Transfer (User Requested).")
    else:
        print("Warning: QNN SDK not found on Host. Assuming Device has all includes/libs in place.")

    for f in files_to_upload:
        if os.path.exists(f):
            size_bytes = os.path.getsize(f)
            print(f"Uploading {os.path.basename(f)} ({size_bytes/1024/1024:.2f} MB)...")
            scp.put(f, remote_path=f"{REMOTE_BASE_DIR}/{os.path.basename(f)}")
        else:
            print(f"Warning: File {f} not found!")
            
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
