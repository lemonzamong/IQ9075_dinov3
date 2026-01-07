
import argparse
import sys
import os
import subprocess
import paramiko
from scp import SCPClient

# Connection Config (Should match deploy.py or be configurable)
DEVICE_IP = "192.168.0.202"
USERNAME = "ubuntu"
PASSWORD = "qualcomm"
REMOTE_BASE_DIR = "/home/ubuntu/dinov3_deployment"

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(server, port, user, password, banner_timeout=200, timeout=10)
        return client
    except Exception as e:
        print(f"Failed to connect to {server}: {e}")
        return None

def run_command(client, command):
    print(f"[REMOTE] {command}")
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out: print(out)
    if err: print(err)
    return exit_status

def progress_bar(filename, size, sent):
    sys.stdout.write(f"\rUploading {filename}: {float(sent)/float(size)*100:.2f}%")
    sys.stdout.flush()
    if sent == size:
        sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Run DINOv3 Inference on IQ-9075 (HTP)")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output_dir", default="inference_results", help="Local directory to save results")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found.")
        return

    # 1. Preprocess Image
    print(f"--- Preprocessing {args.image_path} ---")
    raw_path = "temp_input.raw"
    # Assuming preprocess_input.py is in the same 'scripts' dir or CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(script_dir, "preprocess_input.py")
    
    try:
        subprocess.run([sys.executable, preprocess_script, args.image_path, raw_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed: {e}")
        return

    # 2. Connect
    print(f"--- Connecting to {DEVICE_IP} ---")
    ssh = create_ssh_client(DEVICE_IP, 22, USERNAME, PASSWORD)
    if not ssh: return
    scp = SCPClient(ssh.get_transport(), progress=progress_bar)

    # 3. Upload Input
    remote_input_path = f"{REMOTE_BASE_DIR}/test/custom_input.raw"
    remote_input_list = f"{REMOTE_BASE_DIR}/test/custom_input_list.txt"
    
    print(f"--- Uploading Input ---")
    scp.put(raw_path, remote_input_path)
    os.remove(raw_path) # Clean local temp

    # 4. Create Input List on Device
    run_command(ssh, f"echo 'pixel_values:={remote_input_path}' > {remote_input_list}")

    # 5. Run Execution
    print(f"--- Running Inference (HTP) ---")
    # Using the same environment setup as deploy.py
    cmd = f"cd {REMOTE_BASE_DIR} && " \
          f"export ADSP_LIBRARY_PATH=\"{REMOTE_BASE_DIR}/lib/hexagon;/usr/lib/rfsa/adsp;/dsp;/usr/lib/dsp/cdsp1;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp\" && " \
          f"export LD_LIBRARY_PATH={REMOTE_BASE_DIR}/lib:$LD_LIBRARY_PATH && " \
          f"./bin/qnn-net-run --backend {REMOTE_BASE_DIR}/lib/libQnnHtp.so " \
          f"--model {REMOTE_BASE_DIR}/bin/libdinov3.so " \
          f"--input_list {remote_input_list} " \
          f"--output_dir {REMOTE_BASE_DIR}/test/custom_output"
    
    exit_code = run_command(ssh, cmd)
    
    if exit_code != 0:
        print("Inference failed! Check logs above.")
        return

    # 6. Download Results
    print(f"--- Downloading Results to {args.output_dir} ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tar remote output for efficiency
    run_command(ssh, f"tar -czf {REMOTE_BASE_DIR}/custom_output.tar.gz -C {REMOTE_BASE_DIR}/test/custom_output .")
    try:
        scp.get(f"{REMOTE_BASE_DIR}/custom_output.tar.gz", "custom_output.tar.gz")
        subprocess.run(f"tar -xzf custom_output.tar.gz -C {args.output_dir}", shell=True)
        os.remove("custom_output.tar.gz")
        print(f"Success! Results saved in {args.output_dir}")
    except Exception as e:
        print(f"Download failed: {e}")

    # Cleanup remote temporary tar
    run_command(ssh, f"rm {REMOTE_BASE_DIR}/custom_output.tar.gz")
    ssh.close()

if __name__ == "__main__":
    main()
