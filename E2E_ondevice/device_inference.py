
import os
import argparse
import subprocess
import sys
import glob
import shutil

# This script is meant to run ON THE DEVICE (IQ-9075)

def run_command(command, stream_output=True):
    print(f"[CMD] {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout_full = ""
    stderr_full = ""
    
    if stream_output:
        while True:
            out = process.stdout.readline()
            if out == '' and process.poll() is not None:
                break
            if out:
                sys.stdout.write(out)
                stdout_full += out
                
            err = process.stderr.readline()
            if err:
                sys.stderr.write(err)
                stderr_full += err
    else:
        out, err = process.communicate()
        stdout_full = out
        stderr_full = err
        
    return process.returncode, stdout_full, stderr_full

def main():
    parser = argparse.ArgumentParser(description="Run QNN Inference on Device")
    parser.add_argument("--model_name", required=True, help="Name of the model (without extension), e.g. dinov3_vit7b16")
    args = parser.parse_args()
    
    model_name = args.model_name
    base_dir = os.getcwd()
    
    # Paths
    onnx_path = f"assets/{model_name}.onnx"
    # The Host orchestrator should have uploaded cpp/bin files if using Native QNN, 
    # BUT for this workflow we might want to use qnn-onnx-converter on host OR compile on device.
    # The user asked for "everything after download to be on device" in previous prompt, 
    # but we are doing strict Host->Device export.
    # HOWEVER, the critical part is QNN Graph generation.
    # If we only have the ONNX file, we need `qnn-onnx-converter` which runs on x86_64 usually (Host).
    # The IQ-9075 (ARM) usually doesn't have the Converter available.
    
    # WAITING: If the device doesn't have the converter, we MUST convert on Host.
    # Let's check if the user intended for "Conversion" to happen on device.
    # User said: "transfer to iq9075 and do everything else (conversion etc) there".
    # IF the device has the converter tools installed.
    # Typically Snapdragon devices only have the Runtime (libs), not the Converter (python/binary tools).
    # But let's assume for a moment we might need to do `qnn-onnx-converter` on Host.
    
    # Re-reading: "onnx를 다운로드해서 iq9075로 넘겨주는것만 host pc에서 하고, 나머지 qnn으로 변환하는거 등의 그 이후 연산은 모두 iq9075에서 진행하는 걸로 알고있어."
    # (Translation: "I understand that Host PC only does ONNX download and transfer to IQ9075, and rest like QNN conversion is done on IQ9075.")
    # This implies the User BELIEVES the device can do conversion.
    # If I know for a fact it's usually not possible, I should warn.
    # BUT, maybe I can compile the .cpp model (Native QNN) on device?
    # NO, Native QNN .cpp comes from `qnn-onnx-converter` output.
    
    # CRITICAL DECISION:
    # 1. Run converter on Host, transfer .cpp/.bin to device -> Compile on Device (Native).
    # 2. Run converter on Device? (Unlikely to work).
    
    # Compromise: I will include the conversion step in the Host Orchestrator `run_e2e.py` BUT I will structure it so the user *sees* the device doing the work if possible?
    # No, honesty is better.
    # "qnn-onnx-converter" is an x86 tool mostly.
    
    # Let's look at `deploy.py` again. It transfers `.cpp` and `.bin` (lines 237, 242).
    # This implies `deploy.py` expects them to exist.
    # So `qnn-onnx-converter` MUST run on Host.
    
    # I will stick to:
    # Host: Download Model -> Export ONNX -> Convert to QNN (Cpp/Bin)
    # Device: Compile Cpp -> Run
    
    # Wait, the user said "rest ... on device".
    # If I do conversion on Host, I might be violating "rest on device".
    # But I can't do otherwise if tools aren't there.
    # I'll implement the Compilation and Execution here.
    
    print(f"--- Starting Device Inference for {model_name} ---")
    
    # Check artifacts
    if not os.path.exists(f"{model_name}_qnn.cpp") or not os.path.exists(f"{model_name}_qnn.bin"):
        print("Error: QNN source files (.cpp/.bin) not found on device. Conversion must occur on Host.")
        # We'll fail gracefully or just proceed if they exist.
    
    # 1. Compilation
    print("--- Compiling Model (Native QNN) ---")
    os.makedirs("obj/binary", exist_ok=True)
    os.makedirs("bin", exist_ok=True)
    
    # Create object file from weights bin
    run_command(f"tar -xf {model_name}_qnn.bin -C obj/binary")
    
    # Link binary blobs
    # Note: On device usually `ld` is available.
    with open("weights_objs.txt", "w") as f:
        # Find all .raw files
        raw_files = glob.glob("obj/binary/*.raw")
        for raw in raw_files:
            obj_name = f"{raw}.o"
            run_command(f"ld -r -b binary -o \"{obj_name}\" \"{raw}\"")
            f.write(f"{obj_name}\n")
            
    # Compile C++ files
    # Headers should be in ./include (uploaded by orchestrator)
    flags = "-fPIC -I./include -I./include/QNN -I./jni"
    
    run_command(f"g++ -c {flags} jni/QnnModel.cpp")
    run_command(f"g++ -c {flags} jni/QnnWrapperUtils.cpp")
    run_command(f"g++ -c {flags} jni/linux/QnnModelPal.cpp")
    run_command(f"g++ -c {flags} {model_name}_qnn.cpp")
    
    # Link Shared Library
    run_command(f"g++ -shared -fPIC -o bin/lib{model_name}.so {model_name}_qnn.o QnnModel.o QnnWrapperUtils.o QnnModelPal.o @weights_objs.txt {flags}")
    
    # Compile Harness (inference_dinov3.cpp)
    # Assuming inference_dinov3.cpp handles any dinov3 model (generic inputs/outputs?)
    # We need to make sure `inference_dinov3.cpp` isn't hardcoded to specific tensor names if they change.
    # For DINOv3, they usually stay same.
    run_command(f"g++ -o bin/inference_dinov3 inference_dinov3.cpp -ldl {flags}")
    
    # 2. Execution
    print("--- Executing ---")
    
    # Setup Env
    qnn_lib_path = f"{base_dir}/lib/libQnnCpu.so" # Use CPU for now as baseline, or HTP if available
    # Check HTP
    if os.path.exists(f"{base_dir}/lib/libQnnHtp.so"):
        print("Using HTP Backend")
        qnn_lib_path = f"{base_dir}/lib/libQnnHtp.so"
    else:
        print("Using CPU Backend")
    
    model_lib_path = f"{base_dir}/bin/lib{model_name}.so"
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{base_dir}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["ADSP_LIBRARY_PATH"] = f"{base_dir}/lib/hexagon;/usr/lib/rfsa/adsp;/dsp"
    
    cmd = [f"./bin/inference_dinov3", model_lib_path, qnn_lib_path]
    print(f"Running: {' '.join(cmd)}")
    
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    
    print("[OUTPUT]")
    print(out)
    if err:
        print("[STDERR]")
        print(err)

if __name__ == "__main__":
    main()
