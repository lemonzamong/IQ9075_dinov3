
import paramiko
import time

DEVICE_IP = "192.168.0.202"
USERNAME = "ubuntu"
PASSWORD = "qualcomm"

def check_status():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(DEVICE_IP, username=USERNAME, password=PASSWORD)
        
        print("--- Processes ---")
        stdin, stdout, stderr = client.exec_command("ps -ef | grep python")
        print(stdout.read().decode())
        
        print("--- Disk Check ---")
        stdin, stdout, stderr = client.exec_command("df -h")
        print(stdout.read().decode())
        
        print("--- Exported Files ---")
        stdin, stdout, stderr = client.exec_command("ls -lh dinov3_e2e/assets")
        print(stdout.read().decode())
        
        print("--- Finding Converter ---")
        # Search widely but limit depth to avoid hang
        stdin, stdout, stderr = client.exec_command("find /opt /home -name qnn-onnx-converter -type f 2>/dev/null | head -n 5")
        print(stdout.read().decode())
        
        client.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    check_status()
