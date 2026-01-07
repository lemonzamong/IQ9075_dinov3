import paramiko
from scp import SCPClient
import os
import sys

# Device Configuration
DEVICE_IP = "192.168.0.202"
USERNAME = "ubuntu"
PASSWORD = "qualcomm"
REMOTE_DIR = "/home/ubuntu/onnx_convert"

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def run_command(client, command):
    print(f"[REMOTE CMD] {command}")
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    if out: print(f"[STDOUT]\n{out}")
    if err: print(f"[STDERR]\n{err}")
    
    if exit_status != 0:
        print(f"Command failed with status {exit_status}")
    
    return exit_status

def main():
    print(f"Connecting to {DEVICE_IP}...")
    try:
        ssh = create_ssh_client(DEVICE_IP, 22, USERNAME, PASSWORD)
        # Increase socket timeout for large file transfers
        ssh.get_transport().set_keepalive(30)
        scp = SCPClient(ssh.get_transport(), socket_timeout=600.0) 
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Create remote directory
    run_command(ssh, f"mkdir -p {REMOTE_DIR}")

    # 2. Files to transfer
    files_to_transfer = [
        "inference_dinov3.cpp",
        "CMakeLists.txt",
        "2_run_on_device.sh",
        "inference_qnn_onnx.py",
        "requirements.txt"  # Optional, if they want to run python app
    ]

    print("Transferring files...")
    for f in files_to_transfer:
        if os.path.exists(f):
            scp.put(f, remote_path=f"{REMOTE_DIR}/{f}")
            print(f"Uploaded {f}")
        else:
            print(f"Warning: Local file {f} not found.")

    # Transfer libs if they exist (Assumed output from Step 2)
    if os.path.isdir("libs"):
        print("Transferring libs folder...")
        scp.put("libs", recursive=True, remote_path=REMOTE_DIR)
    else:
        print("Warning: 'libs' folder not found. Step 2 (Conversion) might not have run.")

    # Transfer Model if exists
    if os.path.exists("dinov3.onnx"):
        print("Transferring ONNX model...")
        scp.put("dinov3.onnx", remote_path=f"{REMOTE_DIR}/dinov3.onnx")
        
    if os.path.exists("dinov3.onnx.data"):
        print("Checking dinov3.onnx.data...")
        local_size = os.path.getsize("dinov3.onnx.data")
        uploaded = False
        try:
             # Check remote size using stat
             stdin, stdout, stderr = ssh.exec_command(f"stat -c %s {REMOTE_DIR}/dinov3.onnx.data")
             output = stdout.read().decode().strip()
             if output.isdigit():
                 remote_size = int(output)
                 if local_size == remote_size:
                     print(f"File size matches ({local_size} bytes). Skipping upload.")
                     uploaded = True
                 else:
                     print(f"Size mismatch (Local: {local_size} vs Remote: {remote_size}). Re-uploading...")
             else:
                 print("Remote file not found or stat failed. Uploading...")
        except Exception as e:
             print(f"Check failed ({e}). Uploading...")
        
        if not uploaded:
            scp.put("dinov3.onnx.data", remote_path=f"{REMOTE_DIR}/dinov3.onnx.data")

    # Probe for QNN SDK and Tools
    print("Probing device for QNN SDK...")
    
    # Check for converter
    stdin, stdout, stderr = ssh.exec_command("which qnn-onnx-converter")
    converter_path = stdout.read().decode().strip()
    if converter_path:
        print(f"Found qnn-onnx-converter at: {converter_path}")
        # If converter is found, we can potentially convert on device!
    else:
        print("qnn-onnx-converter not found in PATH.")

    # Check for SDK Root via Library
    # We search common locations to avoid full disk scan hanging
    search_cmd = "find /opt/qcom /usr/lib /home/ubuntu -name 'libQnnCpu.so' 2>/dev/null | head -n 1"
    stdin, stdout, stderr = ssh.exec_command(search_cmd)
    lib_path = stdout.read().decode().strip()
    
    sdk_root = ""
    backend_lib_path = ""
    
    if lib_path:
        print(f"Found libQnnCpu.so at: {lib_path}")
        backend_lib_path = lib_path
        
        # Try to infer SDK root
        if "lib/aarch64" in lib_path:
            sdk_root = lib_path.split("/lib/aarch64")[0]
            print(f"Inferred SDK Root: {sdk_root}")
        elif "/usr/lib" in lib_path:
            # System install
            sdk_root = "/usr"
            print(f"Inferred System Install Root: {sdk_root}")
    else:
        print("Could not locate libQnnCpu.so in standard paths.")

    # Check for Headers
    stdin, stdout, stderr = ssh.exec_command("find /usr/include /opt/qcom -name QnnInterface.h 2>/dev/null | head -n 1")
    header_path = stdout.read().decode().strip()
    
    run_cpp = False
    if header_path:
        print(f"Found QNN Headers at: {header_path}")
        run_cpp = True
    else:
        print("Warning: QNN Headers (QnnInterface.h) not found. Skipping C++ build/inference.")

    # Generate customized run script LOCALLY
    # detailed logic to build the script based on findings
    
    script_content = "#!/bin/bash\n\n"
    script_content += "echo '--- Running Device Script ---'\n"
    
    if sdk_root:
        script_content += f"export QNN_SDK_ROOT='{sdk_root}'\n"
    
    # Run C++ if headers found and backend found
    if run_cpp and backend_lib_path:
        script_content += "echo '--- Step 3: Building Inference Application ---'\n"
        script_content += "mkdir -p build && cd build\n"
        script_content += "cmake ..\n"
        script_content += "make\n"
        script_content += "cd ..\n"
        script_content += "echo '--- Step 4: Running Inference (C++) ---'\n"
        # We need model lib path. Assume it was uploaded to libs/... or just in root if flat?
        # The host script generated libs/aarch64-linux-clang/libdinov3.so
        # If libs folder was uploaded, we use that.
        script_content += f"MODEL_LIB='./libs/aarch64-linux-clang/libdinov3.so'\n"
        script_content += f"BACKEND_LIB='{backend_lib_path}'\n"
        script_content += "if [ -f \"$MODEL_LIB\" ]; then\n"
        script_content += "   ./build/inference_dinov3 \"$MODEL_LIB\" \"$BACKEND_LIB\"\n"
        script_content += "else\n"
        script_content += "   echo 'Model lib not found (Did conversion run on host?)'\n"
        script_content += "fi\n"

    # Run Python if available
    script_content += "\necho '--- Checking for Python Inference ---'\n"
    script_content += "if command -v python3 &> /dev/null; then\n"
    script_content += "   echo 'Python3 found.'\n"
    
    # Try to install dependencies in venv to avoid PEP 668 issues
    script_content += "   echo 'Creating virtual environment...'\n"
    script_content += "   python3 -m venv venv\n"
    script_content += "   source venv/bin/activate\n"
    script_content += "   echo 'Installing Python dependencies (onnxruntime, numpy)...'\n"
    script_content += "   pip install onnxruntime numpy\n"

    # Construct python command
    # If backend_lib_path found, usage: --backend qnn_cpu --qnn_lib <path>
    if backend_lib_path:
        py_cmd = f"python inference_qnn_onnx.py --model dinov3.onnx --backend qnn_cpu --qnn_lib {backend_lib_path}"
    else:
        py_cmd = "python inference_qnn_onnx.py --model dinov3.onnx --backend cpu"

    script_content += f"   echo 'Running: {py_cmd}'\n"
    script_content += f"   {py_cmd}\n"
    script_content += "else\n"
    script_content += "   echo 'Python3 not found.'\n"
    script_content += "fi\n"

    # Write local temp script
    with open("2_run_on_device_dyn.sh", "w") as f:
        f.write(script_content)
    
    # Upload dynamic script
    print("Uploading customized run script...")
    scp.put("2_run_on_device_dyn.sh", remote_path=f"{REMOTE_DIR}/2_run_on_device_dyn.sh")
    run_command(ssh, f"chmod +x {REMOTE_DIR}/2_run_on_device_dyn.sh")

    # Run
    print("Executing device script...")
    run_command(ssh, f"cd {REMOTE_DIR} && ./2_run_on_device_dyn.sh")

    scp.close()
    ssh.close()
    print("Deployment and execution sequence finished.")

if __name__ == "__main__":
    main()


