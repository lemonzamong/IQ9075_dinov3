import paramiko
from scp import SCPClient
import os
import sys
import argparse
import subprocess
import glob
import shutil
import hashlib

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


def cleanup_temp_files():
    """Clean up temporary files created during deployment."""
    temp_files = ["sdk_headers.tar.gz", "sdk_jni.tar.gz", "sdk_libs.tar.gz", "run_on_device.sh", "input_list.txt", "skel_libs.tar.gz", "output.tar.gz"]
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

def get_local_md5(path):
    print(f"Calculating local MD5 for {os.path.basename(path)}...", end="", flush=True)
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            hash_md5.update(chunk)
    digest = hash_md5.hexdigest()
    print(" Done.")
    return digest

def transfer_file_smart(ssh, scp, local_path, remote_path, check_integrity=True):
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
                # Size matched, optionally check MD5 for robustness
                if check_integrity and local_size > 0: # Check MD5
                     print(f"[{filename}] Size matches. Verifying MD5...")
                     stdin, stdout, stderr = ssh.exec_command(f"md5sum {remote_path}")
                     remote_md5 = stdout.read().decode().split()[0]
                     local_md5_val = get_local_md5(local_path)
                     
                     if remote_md5 == local_md5_val:
                         print(f"[{filename}] Integrity Confirmed. Skipping.")
                         return True
                     else:
                         print(f"[{filename}] MD5 Mismatch ({local_md5_val} vs {remote_md5}). Re-uploading.")
                else:
                     print(f"[{filename}] Size matches ({local_size} bytes). Skipping.")
                     return True
    except Exception as e:
        # print(f"Check failed: {e}")
        pass

    print(f"[{filename}] Uploading...")
    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote_path)
    run_command(ssh, f"mkdir -p {remote_dir}", stream_output=False)
    scp.put(local_path, remote_path=remote_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy and Run on IQ-9075")
    parser = argparse.ArgumentParser(description="Deploy and Run on IQ-9075 (Native QNN Only)")
    # Removed --mode, defaulting to native
    
    print(f"Connecting to {DEVICE_IP}...")
    try:
        ssh = create_ssh_client(DEVICE_IP, 22, USERNAME, PASSWORD)
        ssh.get_transport().set_keepalive(30)
        scp = SCPClient(ssh.get_transport(), socket_timeout=600.0, progress=progress_bar)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Cleanup Old Directory on Device
    print("--- Cleaning up Legacy Directories ---")
    run_command(ssh, "rm -rf /home/ubuntu/onnx_convert", stream_output=True)

    # 1. Create remote directory structure
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/bin")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/assets")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/test")

    # 1. Create remote directory structure
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/bin")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/assets")
    run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/test")

    # 2. Probe Device for SDK/Libs
    print("Probing device environment...")
    sdk_root = ""
    backend_lib_path = ""
    
    # Find libQnnCpu.so - Check bundled first, then system
    _, lib_path, _ = run_command(ssh, f"find {REMOTE_BASE_DIR}/lib /opt/qcom /usr/lib /home/ubuntu -name 'libQnnCpu.so' 2>/dev/null | head -n 1", stream_output=False)
    if lib_path:
        # Prefer the one we deploy if available
        if REMOTE_BASE_DIR in lib_path:
             print(f"Found Bundled Backend: {lib_path}")
             backend_lib_path = lib_path
        else:
             print(f"Found System Backend: {lib_path}")
             backend_lib_path = lib_path
             if "lib/aarch64" in lib_path:
                 sdk_root = lib_path.split("/lib/aarch64")[0]
             elif "/usr/lib" in lib_path:
                 sdk_root = "/usr"
    else:
        print("Warning: libQnnCpu.so not found. Will attempt to upload.")

    # 3. Transfer Assets (Model, Configs)
    print("--- Syncing Assets ---")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3.onnx", f"{REMOTE_BASE_DIR}/assets/dinov3.onnx")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3.onnx.data", f"{REMOTE_BASE_DIR}/assets/dinov3.onnx.data")
    transfer_file_smart(ssh, scp, f"{DIR_ASSETS}/dinov3_qnn_net.json", f"{REMOTE_BASE_DIR}/assets/dinov3_qnn_net.json")

    # Probe for HTP Backend
    print("Probing for HTP Backend...")
    # Typically libQnnHtp.so or libQnnHtpV73.so
    _, lib_path, _ = run_command(ssh, f"find {REMOTE_BASE_DIR}/lib /opt/qcom /usr/lib /home/ubuntu -name 'libQnnHtp.so' 2>/dev/null | head -n 1", stream_output=False)
    backend_lib_path = ""
    if lib_path:
        print(f"Found HTP Backend: {lib_path}")
        backend_lib_path = lib_path
        if "lib/aarch64" in lib_path:
             sdk_root = lib_path.split("/lib/aarch64")[0]
        elif "/usr/lib" in lib_path:
             sdk_root = "/usr"
    else:
        print("Warning: libQnnHtp.so not found. System might default to CPU or similar. Will look for CPU as fallback to ensure something runs, but HTP is requested.")
        _, lib_path_cpu, _ = run_command(ssh, f"find {REMOTE_BASE_DIR}/lib /opt/qcom /usr/lib /home/ubuntu -name 'libQnnCpu.so' 2>/dev/null | head -n 1", stream_output=False)
        if lib_path_cpu:
             print(f"Falling back to CPU Backend: {lib_path_cpu}")
             backend_lib_path = lib_path_cpu
        else:
             print("Error: No QNN backend found.")
             return

    # 4. Generate Script
    script_content = "#!/bin/bash\n\n"
    script_content += f"cd {REMOTE_BASE_DIR}\n"
    script_content += "echo '--- Starting Device Execution ---'\n"

    # Native QNN Mode Logic
    qnn_sdk_host = os.environ.get("QNN_SDK_ROOT", "")
    if not qnn_sdk_host:
            possible_paths = ["/home/hyeokjun/IQ-9075 Evaluation Kit (EVK)/v2.41.0.251128/qairt/2.41.0.251128"]
            for p in possible_paths:
                if os.path.isdir(p):
                    qnn_sdk_host = p
                    break
        
    # Check if we already have processed weights
    has_processed_weights = False
    _, out, _ = run_command(ssh, f"[ -f {REMOTE_BASE_DIR}/weights_objs.txt ] && echo 'yes' || echo 'no'", stream_output=False)
    if out == "yes":
        print("Found pre-processed weights (weights_objs.txt). Skipping .bin upload and extraction.")
        has_processed_weights = True
    
    # Upload Source (Small)
    transfer_file_smart(ssh, scp, f"{DIR_NATIVE_SRC}/dinov3_qnn.cpp", f"{REMOTE_BASE_DIR}/dinov3_qnn.cpp")
    transfer_file_smart(ssh, scp, f"{DIR_NATIVE_SRC}/inference_dinov3.cpp", f"{REMOTE_BASE_DIR}/inference_dinov3.cpp")
    
    # Upload Weights (.bin) ONLY if we don't have processed weights
    if not has_processed_weights:
         transfer_file_smart(ssh, scp, f"{DIR_NATIVE_BIN}/dinov3_qnn.bin", f"{REMOTE_BASE_DIR}/dinov3_qnn.bin")

    # Compile and Link (On-Device)
    print("--- Checking Dependencies ---")
    
    # Upload Headers (Tar optimized)
    if qnn_sdk_host:
         subprocess.run(f"tar -czf sdk_headers.tar.gz -C \"{qnn_sdk_host}\" include", shell=True, check=True)
         if transfer_file_smart(ssh, scp, "sdk_headers.tar.gz", f"{REMOTE_BASE_DIR}/sdk_headers.tar.gz"):
             run_command(ssh, f"tar -xzf {REMOTE_BASE_DIR}/sdk_headers.tar.gz -C {REMOTE_BASE_DIR} && rm {REMOTE_BASE_DIR}/sdk_headers.tar.gz")
         
         # Upload JNI (Tar optimized)
         share_jni_path = f"{qnn_sdk_host}/share/QNN/converter/jni"
         subprocess.run(f"tar -czf sdk_jni.tar.gz -C \"{os.path.dirname(share_jni_path)}\" jni", shell=True, check=True)
         if transfer_file_smart(ssh, scp, "sdk_jni.tar.gz", f"{REMOTE_BASE_DIR}/sdk_jni.tar.gz"):
             run_command(ssh, f"tar -xzf {REMOTE_BASE_DIR}/sdk_jni.tar.gz -C {REMOTE_BASE_DIR} && rm {REMOTE_BASE_DIR}/sdk_jni.tar.gz")

    # Build Script
    script_content += "echo '--- Compiling ---'\n"
    
    # Skip extraction if we have weights
    if has_processed_weights:
         script_content += "echo 'Using existing processed weights...'\n"
    else:
         script_content += "mkdir -p obj/binary\n"
         script_content += "tar -xf dinov3_qnn.bin -C obj/binary\n"
         script_content += "find obj/binary -name '*.raw' | while read f; do ld -r -b binary -o \"$f.o\" \"$f\"; done\n"
         script_content += "find obj/binary -name '*.raw.o' > weights_objs.txt\n"
    
    script_content += "g++ -c -fPIC jni/QnnModel.cpp -I./include -I./include/QNN -I./jni\n"
    script_content += "g++ -c -fPIC jni/QnnWrapperUtils.cpp -I./include -I./include/QNN -I./jni\n"
    script_content += "g++ -c -fPIC jni/linux/QnnModelPal.cpp -I./include -I./include/QNN -I./jni\n"
    script_content += "g++ -c -fPIC dinov3_qnn.cpp -I./include -I./include/QNN -I./jni\n"
    
    script_content += "g++ -shared -fPIC -o bin/libdinov3.so dinov3_qnn.o QnnModel.o QnnWrapperUtils.o QnnModelPal.o @weights_objs.txt -I./include -I./include/QNN -I./jni\n"
    
    script_content += "g++ -o bin/inference_dinov3 inference_dinov3.cpp -ldl -I./include -I./include/QNN -I./jni\n"
    script_content += f"export LD_LIBRARY_PATH={REMOTE_BASE_DIR}/lib:$LD_LIBRARY_PATH\n"
    script_content += f"./bin/inference_dinov3 {REMOTE_BASE_DIR}/bin/libdinov3.so {REMOTE_BASE_DIR}/lib/libQnnCpu.so\n"
    
    # Upload Libs (Smart Tar)
    print("--- Syncing SDK Libraries (OpenEmbedded GCC 11.2) ---")
    # Using OE GCC 11.2 as recommended for IQ9075
    target_arch = "aarch64-oe-linux-gcc11.2"
    
    # Check if libs exist remotely (force update if we are changing targets, but smart check should handle it if size differs)
    # Actually, we should probably force update since we are changing targets
    subprocess.run(f"tar -czf sdk_libs.tar.gz -C \"{qnn_sdk_host}/lib/{target_arch}\" .", shell=True, check=True)
    if transfer_file_smart(ssh, scp, "sdk_libs.tar.gz", f"{REMOTE_BASE_DIR}/sdk_libs.tar.gz"):
        print("Extracting libs on device...")
        run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib && tar -xzf {REMOTE_BASE_DIR}/sdk_libs.tar.gz -C {REMOTE_BASE_DIR}/lib && rm {REMOTE_BASE_DIR}/sdk_libs.tar.gz")
    
    # Upload qnn-net-run
    qnn_net_run_src = f"{qnn_sdk_host}/bin/{target_arch}/qnn-net-run"
    transfer_file_smart(ssh, scp, qnn_net_run_src, f"{REMOTE_BASE_DIR}/bin/qnn-net-run")
    run_command(ssh, f"chmod +x {REMOTE_BASE_DIR}/bin/qnn-net-run")

    # Upload Hexagon Skel Libs (Fix for CRC mismatch)
    print("--- Syncing Hexagon Skel Libraries ---")
    # Finding all Skel.so files in local SDK
    skel_dirs = glob.glob(f"{qnn_sdk_host}/lib/hexagon-v*/unsigned/*.so")
    
    os.makedirs("temp_skel", exist_ok=True)
    for skel_file in skel_dirs:
        shutil.copy(skel_file, "temp_skel/")
    
    subprocess.run("tar -czf skel_libs.tar.gz -C temp_skel .", shell=True, check=True)
    if transfer_file_smart(ssh, scp, "skel_libs.tar.gz", f"{REMOTE_BASE_DIR}/skel_libs.tar.gz"):
        print("Extracting Skel libs on device...")
        run_command(ssh, f"mkdir -p {REMOTE_BASE_DIR}/lib/hexagon && tar -xzf {REMOTE_BASE_DIR}/skel_libs.tar.gz -C {REMOTE_BASE_DIR}/lib/hexagon && rm {REMOTE_BASE_DIR}/skel_libs.tar.gz")
    
    # Clean up local temp
    shutil.rmtree("temp_skel")
    if os.path.exists("skel_libs.tar.gz"): os.remove("skel_libs.tar.gz")

    # Verify
    if os.path.exists(f"{DIR_TEST}/test_image.jpg"):
         print("--- Preprocessing Test Image ---")
         subprocess.run([sys.executable, f"{DIR_SCRIPTS}/preprocess_input.py", f"{DIR_TEST}/test_image.jpg", f"{DIR_TEST}/input.raw"], check=True)
         transfer_file_smart(ssh, scp, f"{DIR_TEST}/input.raw", f"{REMOTE_BASE_DIR}/test/input.raw")
         transfer_file_smart(ssh, scp, f"{DIR_TEST}/input_list.txt", f"{REMOTE_BASE_DIR}/test/input_list.txt") 
         with open("input_list.txt", "w") as f:
             f.write(f"pixel_values:={REMOTE_BASE_DIR}/test/input.raw\n")
         transfer_file_smart(ssh, scp, "input_list.txt", f"{REMOTE_BASE_DIR}/test/input_list.txt")

         script_content += "echo '--- Verifying with qnn-net-run (HTP Backend) ---'\n"
         # Set ADSP_LIBRARY_PATH to our uploaded Skels FIRST
         dsp_lib_path = f"{REMOTE_BASE_DIR}/lib/hexagon"
         script_content += f"export ADSP_LIBRARY_PATH=\"{dsp_lib_path};/usr/lib/rfsa/adsp;/dsp;/usr/lib/dsp/cdsp1;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp\"\n"
         script_content += f"export LD_LIBRARY_PATH={REMOTE_BASE_DIR}/lib:$LD_LIBRARY_PATH\n"
         script_content += f"./bin/qnn-net-run --backend {REMOTE_BASE_DIR}/lib/libQnnHtp.so --model {REMOTE_BASE_DIR}/bin/libdinov3.so --input_list {REMOTE_BASE_DIR}/test/input_list.txt --output_dir {REMOTE_BASE_DIR}/test/output\n"

    # Execute
    with open("run_on_device.sh", "w") as f:
        f.write(script_content)
    
    transfer_file_smart(ssh, scp, "run_on_device.sh", f"{REMOTE_BASE_DIR}/run_on_device.sh")
    run_command(ssh, f"chmod +x {REMOTE_BASE_DIR}/run_on_device.sh")
    
    print("--- Executing Remote Script ---")
    channel = ssh.get_transport().open_session()
    channel.exec_command(f"{REMOTE_BASE_DIR}/run_on_device.sh")
    
    output_buffer = ""
    while True:
        if channel.exit_status_ready(): break
        if channel.recv_ready(): 
            chunk = channel.recv(1024).decode()
            sys.stdout.write(chunk)
            output_buffer += chunk
        if channel.recv_stderr_ready(): 
            err_chunk = channel.recv_stderr(1024).decode()
            sys.stderr.write(err_chunk)
            output_buffer += err_chunk
    
    print("\n\n--- Performance Report ---")
    # Parse for "Avg" or "Inference time" depending on qnn-net-run version
    # Standard qnn-net-run might print "NetRun: ... Avg: X us" or similar.
    # If not found, user can check logs.
    import re
    
    # Download Results
    print(f"\n--- Downloading Results from {REMOTE_BASE_DIR}/test/output ---")
    local_output_dir = "output_results"
    os.makedirs(local_output_dir, exist_ok=True)
    try:
        # Tar remote output folder for faster transfer
        run_command(ssh, f"tar -czf {REMOTE_BASE_DIR}/output.tar.gz -C {REMOTE_BASE_DIR}/test output", stream_output=False)
        scp.get(f"{REMOTE_BASE_DIR}/output.tar.gz", "output.tar.gz")
        subprocess.run(f"tar -xzf output.tar.gz -C {local_output_dir}", shell=True)
        os.remove("output.tar.gz")
        print(f"Results downloaded to: {os.path.abspath(local_output_dir)}")
    except Exception as e:
        print(f"Failed to download results: {e}")
    # Pattern: "NetRun: .* Avg: ([0-9]+) us"
    # Or "Total inference time: ..."
    matches = re.findall(r"Avg: ([0-9\.]+) us", output_buffer)
    if matches:
        print(f"Detected Average Inference Time: {matches[-1]} us ({float(matches[-1])/1000.0:.2f} ms)")
    else:
        print("Could not automatically parse inference time from output. Please check above logs.")

    print("\nDone.")
    cleanup_temp_files()
    scp.close()
    ssh.close()

if __name__ == "__main__":
    main()
