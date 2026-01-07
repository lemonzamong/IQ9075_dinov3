import paramiko
from scp import SCPClient
import os
import sys
import argparse
import subprocess

# Device Configuration
DEVICE_IP = "192.168.0.202"
USERNAME = "ubuntu"
PASSWORD = "qualcomm"
REMOTE_BASE_DIR = "/home/ubuntu/dinov3_deployment"

# Local Paths (New Structure - Relative to onnx_convert/)
DIR_ASSETS = "assets"
DIR_NATIVE_SRC = "native_qnn/src"
DIR_NATIVE_BIN = "native_qnn/bin"
DIR_SCRIPTS = "scripts"
DIR_TEST = "test"
# ORT script is now in scripts/
DIR_ORT_SCRIPT = "scripts/inference_ort.py"

def cleanup_temp_files():
    """Clean up temporary files created during deployment."""
    temp_files = ["sdk_headers.tar.gz", "sdk_jni.tar.gz", "sdk_libs.tar.gz", "run_on_device.sh", "input_list.txt"]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    print("Temporary files cleaned up.")

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # High latency network (7s+ ping), increasing timeouts significantly
    client.connect(server, port, user, password, banner_timeout=200, timeout=200)
    return client

def run_command(client, command, stream_output=True):
    print(f"[REMOTE CMD] {command}")
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    if stream_output:
        if out: print(f"[STDOUT]\n{out}")
        if err: print(f"[STDERR]\n{err}")
    
    if exit_status != 0:
        print(f"Command failed with status {exit_status}")
    
    return exit_status, out, err

# Progress Callback
def progress_bar(filename, size, sent):
    sys.stdout.write(f"\rUploading {filename}: {float(sent)/float(size)*100:.2f}%")
    sys.stdout.flush()
    if sent == size:
        sys.stdout.write('\n')

def transfer_file_smart(ssh, scp, local_path, remote_path):
    if not os.path.exists(local_path):
        print(f"Warning: Local file {local_path} not found.")
        return False

    local_size = os.path.getsize(local_path)
    filename = os.path.basename(local_path)
    
    try:
        stdin, stdout, stderr = ssh.exec_command(f"stat -c %s {remote_path}")
        output = stdout.read().decode().strip()
        if output.isdigit():
            remote_size = int(output)
            if local_size == remote_size:
                print(f"[{filename}] Size matches ({local_size} bytes). Skipping.")
                return True
    except:
        pass

    print(f"[{filename}] Uploading...")
    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote_path)
    run_command(ssh, f"mkdir -p {remote_dir}", stream_output=False)
    scp.put(local_path, remote_path=remote_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy and Run on IQ-9075")
    parser.add_argument("--mode", choices=["ort", "native"], default="native", help="Inference mode")
    args = parser.parse_args()

    print(f"Connecting to {DEVICE_IP} for mode: {args.mode.upper()}...")
    try:
        ssh = create_ssh_client(DEVICE_IP, 22, USERNAME, PASSWORD)
        ssh.get_transport().set_keepalive(30)
        scp = SCPClient(ssh.get_transport(), socket_timeout=600.0, progress=progress_bar)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Create remote directory structure
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/bin")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/src")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/obj")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/assets")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/test")

    # --- REORGANIZATION / REUSE LOGIC ---
    print("--- Checking for existing files to reuse ---")
    
    # 0. Cleanup loose files in REMOTE_BASE_DIR (Move them to folders)
    run_command(ssh, f"mv {REMOTE_BASE_DIR}/*.o {REMOTE_BASE_DIR}/obj/ 2>/dev/null")
    run_command(ssh, f"mv {REMOTE_BASE_DIR}/*.cpp {REMOTE_BASE_DIR}/src/ 2>/dev/null")
    run_command(ssh, f"mv {REMOTE_BASE_DIR}/*.bin {REMOTE_BASE_DIR}/bin/ 2>/dev/null")
    run_command(ssh, f"mv {REMOTE_BASE_DIR}/weights_objs.txt {REMOTE_BASE_DIR}/obj/ 2>/dev/null")

    # 1. Check old legacy dir for migration
    old_dir = "/home/ubuntu/onnx_convert"
    if run_command(ssh, f"[ -d {old_dir} ]", stream_output=False)[0] == 0:
        print(f"Found old directory {old_dir}. Moving reusable files...")
        
        # Move Libs
        run_command(ssh, f"[ -d {old_dir}/lib ] && cp -rn {old_dir}/lib/* {REMOTE_BASE_DIR}/lib/ 2>/dev/null")
        
        # Move Object Files
        run_command(ssh, f"[ -d {old_dir}/obj ] && cp -rn {old_dir}/obj/* {REMOTE_BASE_DIR}/obj/ 2>/dev/null")
        run_command(ssh, f"[ -f {old_dir}/weights_objs.txt ] && cp -n {old_dir}/weights_objs.txt {REMOTE_BASE_DIR}/obj/ 2>/dev/null")
        
        # Move Source/Bin
        run_command(ssh, f"[ -f {old_dir}/dinov3_qnn.cpp ] && cp -n {old_dir}/dinov3_qnn.cpp {REMOTE_BASE_DIR}/src/ 2>/dev/null")
        run_command(ssh, f"[ -f {old_dir}/dinov3_qnn.bin ] && cp -n {old_dir}/dinov3_qnn.bin {REMOTE_BASE_DIR}/bin/ 2>/dev/null")
        
        # Move Assets
        run_command(ssh, f"[ -f {old_dir}/dinov3_qnn_net.json ] && cp -n {old_dir}/dinov3_qnn_net.json {REMOTE_BASE_DIR}/assets/ 2>/dev/null")
        
        print("Restructuring from old directory done.")

    # 2. Probe Device for SDK/Libs
    print("Probing device environment...")
    sdk_root = ""
    backend_lib_path = ""
    
    # Find libQnnCpu.so - Check bundled first, then system
    _, lib_path, _ = run_command(ssh, f"find {REMOTE_BASE_DIR}/lib /opt/qcom /usr/lib /home/ubuntu -name 'libQnnCpu.so' 2>/dev/null | head -n 1", stream_output=False)
    if lib_path:
        if REMOTE_BASE_DIR in lib_path:
             print(f"Found Bundled Backend: {lib_path}")
             backend_lib_path = lib_path
        else:
             print(f"Found System Backend: {lib_path}")
             backend_lib_path = lib_path
    else:
        print("Warning: libQnnCpu.so not found.")

    # 3. Transfer Assets (Model, Configs)
    print("--- Syncing Assets ---")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3.onnx", f"{REMOTE_BASE_DIR}/assets/dinov3.onnx")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3.onnx.data", f"{REMOTE_BASE_DIR}/assets/dinov3.onnx.data")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3_qnn_net.json", f"{REMOTE_BASE_DIR}/assets/dinov3_qnn_net.json")

    # 4. Generate Script
    script_content = "#!/bin/bash\n\n"
    script_content += f"cd {REMOTE_BASE_DIR}\n"
    script_content += "echo '--- Starting Device Execution ---'\n"

    if args.mode == "ort":
        print("--- Syncing ORT Script ---")
        transfer_file_smart(ssh, scp, DIR_ORT_SCRIPT, f"{REMOTE_BASE_DIR}/scripts/inference_ort.py")
        
        script_content += "echo '--- Running ORT Inference ---'\n"
        script_content += "export LD_LIBRARY_PATH=~/dinov3_deployment/lib:$LD_LIBRARY_PATH\n"
        script_content += "python3 ./scripts/inference_ort.py\n"

    elif args.mode == "native":
        qnn_sdk_host = os.environ.get("QNN_SDK_ROOT", "/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/v2.41.0.251128/qairt/2.41.0.251128")
        
        # Check if we already have processed weights
        has_processed_weights = False
        _, out, _ = run_command(ssh, f"[ -f {REMOTE_BASE_DIR}/obj/weights_objs.txt ] && echo 'yes' || echo 'no'", stream_output=False)
        if out == "yes":
            print("Found pre-processed weights. Skipping upload/extract.")
            has_processed_weights = True
        
        # Upload Source
        transfer_file_smart(ssh, scp, f"{DIR_NATIVE_SRC}/dinov3_qnn.cpp", f"{REMOTE_BASE_DIR}/src/dinov3_qnn.cpp")
        transfer_file_smart(ssh, scp, f"{DIR_NATIVE_SRC}/inference_dinov3.cpp", f"{REMOTE_BASE_DIR}/src/inference_dinov3.cpp")
        
        if not has_processed_weights:
             transfer_file_smart(ssh, scp, f"{DIR_NATIVE_BIN}/dinov3_qnn.bin", f"{REMOTE_BASE_DIR}/bin/dinov3_qnn.bin")

        # Sync Headers/JNI
        print("--- Checking Dependencies ---")
        if os.path.isdir(qnn_sdk_host):
             subprocess.run(f"tar -czf sdk_headers.tar.gz -C \"{qnn_sdk_host}\" include", shell=True, check=True)
             if transfer_file_smart(ssh, scp, "sdk_headers.tar.gz", f"{REMOTE_BASE_DIR}/sdk_headers.tar.gz"):
                 run_command(ssh, f"tar -xzf {REMOTE_BASE_DIR}/sdk_headers.tar.gz -C {REMOTE_BASE_DIR} && rm {REMOTE_BASE_DIR}/sdk_headers.tar.gz")
             
             share_jni_path = f"{qnn_sdk_host}/share/QNN/converter/jni"
             subprocess.run(f"tar -czf sdk_jni.tar.gz -C \"{os.path.dirname(share_jni_path)}\" jni", shell=True, check=True)
             if transfer_file_smart(ssh, scp, "sdk_jni.tar.gz", f"{REMOTE_BASE_DIR}/sdk_jni.tar.gz"):
                 run_command(ssh, f"tar -xzf {REMOTE_BASE_DIR}/sdk_jni.tar.gz -C {REMOTE_BASE_DIR} && rm {REMOTE_BASE_DIR}/sdk_jni.tar.gz")

        # Build Script
        script_content += "echo '--- Compiling Native QNN ---'\n"
        if not has_processed_weights:
             script_content += "mkdir -p obj/binary\n"
             script_content += "echo 'Extracting weights...'\n"
             script_content += "tar -xf bin/dinov3_qnn.bin -C obj/binary\n"
             script_content += "echo 'Converting weights to object files...'\n"
             script_content += "find obj/binary -name '*.raw' | while read f; do ld -r -b binary -o \"$f.o\" \"$f\"; done\n"
             script_content += "find obj/binary -name '*.raw.o' > obj/weights_objs.txt\n"
        
        script_content += "echo 'Compiling SDK wrappers...'\n"
        script_content += "g++ -c -fPIC jni/QnnModel.cpp -o obj/QnnModel.o -I./include -I./include/QNN -I./jni\n"
        script_content += "g++ -c -fPIC jni/QnnWrapperUtils.cpp -o obj/QnnWrapperUtils.o -I./include -I./include/QNN -I./jni\n"
        script_content += "g++ -c -fPIC jni/linux/QnnModelPal.cpp -o obj/QnnModelPal.o -I./include -I./include/QNN -I./jni\n"
        script_content += "g++ -c -fPIC src/dinov3_qnn.cpp -o obj/dinov3_qnn.o -I./include -I./include/QNN -I./jni\n"
        
        script_content += "echo 'Linking model library...'\n"
        script_content += "g++ -shared -fPIC -o bin/libdinov3.so obj/dinov3_qnn.o obj/QnnModel.o obj/QnnWrapperUtils.o obj/QnnModelPal.o @obj/weights_objs.txt -I./include -I./include/QNN -I./jni\n"
        
        script_content += "echo 'Compiling inference app...'\n"
        script_content += "rm -f bin/inference_dinov3\n"
        script_content += "g++ -o bin/inference_dinov3 src/inference_dinov3.cpp obj/QnnModel.o obj/QnnWrapperUtils.o obj/QnnModelPal.o -ldl -I./include -I./include/QNN -I./jni\n"
        
        # Test Run
        script_content += "echo '--- Running Verification App ---'\n"
        script_content += f"export LD_LIBRARY_PATH={REMOTE_BASE_DIR}/lib:$LD_LIBRARY_PATH\n"
        # Force System Backend for platform compatibility
        script_content += f"./bin/inference_dinov3 ./bin/libdinov3.so /usr/lib/libQnnCpu.so ./test/input.raw\n"
        
        # Sync SDK Libs
        print("--- Syncing SDK Libraries ---")
        _, out, _ = run_command(ssh, f"[ -f {REMOTE_BASE_DIR}/lib/libQnnCpu.so ] && echo 'yes' || echo 'no'", stream_output=False)
        if out == "yes":
             print("SDK libraries found. Skipping.")
        else:
            subprocess.run(f"tar -czf sdk_libs.tar.gz -C \"{qnn_sdk_host}/lib/aarch64-ubuntu-gcc9.4\" .", shell=True, check=True)
            if transfer_file_smart(ssh, scp, "sdk_libs.tar.gz", f"{REMOTE_BASE_DIR}/sdk_libs.tar.gz"):
                run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib && tar -xzf {REMOTE_BASE_DIR}/sdk_libs.tar.gz -C {REMOTE_BASE_DIR}/lib && rm {REMOTE_BASE_DIR}/sdk_libs.tar.gz")
        
        # qnn-net-run
        transfer_file_smart(ssh, scp, f"{qnn_sdk_host}/bin/aarch64-ubuntu-gcc9.4/qnn-net-run", f"{REMOTE_BASE_DIR}/bin/qnn-net-run")
        run_command(ssh, f"chmod +x {REMOTE_BASE_DIR}/bin/qnn-net-run")

        # Prep Test
        if os.path.exists(f"{DIR_TEST}/test_image.jpg"):
             print("--- Preprocessing Test Image ---")
             subprocess.run([sys.executable, f"{DIR_SCRIPTS}/preprocess_input.py", f"{DIR_TEST}/test_image.jpg", f"{DIR_TEST}/input.raw"], check=True)
             transfer_file_smart(ssh, scp, f"{DIR_TEST}/input.raw", f"{REMOTE_BASE_DIR}/test/input.raw")
             with open("input_list.txt", "w") as f:
                 f.write(f"pixel_values:={REMOTE_BASE_DIR}/test/input.raw\n")
             transfer_file_smart(ssh, scp, "input_list.txt", f"{REMOTE_BASE_DIR}/test/input_list.txt")

    # Final Execute
    with open("run_on_device.sh", "w") as f:
        f.write(script_content)
    transfer_file_smart(ssh, scp, "run_on_device.sh", f"{REMOTE_BASE_DIR}/run_on_device.sh")
    run_command(ssh, f"chmod +x {REMOTE_BASE_DIR}/run_on_device.sh")
    
    print("--- Executing Remote Script ---")
    stdin, stdout, stderr = ssh.exec_command(f"{REMOTE_BASE_DIR}/run_on_device.sh")
    
    # Read output line by line as it comes
    while True:
        line = stdout.readline()
        if not line: break
        sys.stdout.write(line)
        sys.stdout.flush()
    
    err = stderr.read().decode().strip()
    if err:
        sys.stderr.write(f"\n[STDERR]\n{err}\n")
    
    print("\nDone.")
    cleanup_temp_files()
    scp.close()
    ssh.close()

if __name__ == "__main__":
    main()
