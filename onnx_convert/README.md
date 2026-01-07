# Dino v3 QNN Deployment on IQ-9075

This project provides a complete workflow for deploying the Dino v3 model on the Qualcomm IQ-9075 EVK using both **ONNX Runtime (QNN EP)** and **Native QNN (C++ API)**.

## 📂 Project Structure

### Host (Local Machine)
*   `assets/`: Model artifacts (`dinov3.onnx`, `dinov3_qnn_net.json`).
*   `native_qnn/src/`: C++ inference source code.
*   `native_qnn/bin/`: Large pre-processed weights (`dinov3_qnn.bin`).
*   `scripts/`: Deployment and preprocessing Python scripts.
*   `test/`: Test images and raw input data.

### Device (IQ-9075)
Target Directory: `~/dinov3_deployment`
*   `bin/`: Compiled executables and shared model libraries.
*   `lib/`: QNN SDK runtime libraries (Bundled & System).
*   `src/`: C++ source files for on-device compilation.
*   `obj/`: Intermediate object files and processed weights.
*   `assets/`: Model JSON configurations.
*   `test/`: Raw input images and output tensors.

---

## 🚀 Quick Start (Automated Deployment)

To deploy everything and run a verification test in one command:

```bash
# For Native QNN Mode (Performance Mode)
python scripts/deploy.py --mode native

# For ONNX Runtime Mode (Compatibility Mode)
python scripts/deploy.py --mode ort
```

---

## 🛠 Manual Execution via SSH

If you want to run the tests directly on the device console:

### 1. Connect to Device
```bash
ssh ubuntu@192.168.0.202
# Password: qualcomm
```

### 2. Set Environment & Execute
```bash
cd ~/dinov3_deployment

# 1. Set the library path to include our libs and system libs
export LD_LIBRARY_PATH=~/dinov3_deployment/lib:$LD_LIBRARY_PATH

# 2. Run the Hardware Health Check
# This verifies that the model, backend, and hardware are correctly linked.
./bin/inference_dinov3 ./bin/libdinov3.so /usr/lib/libQnnCpu.so
```

### 3. Visibility (Is it working?)
When you run the command above, look for these specific lines:
*   `[ SUCCESS ] Qualcomm Hardware Backend Initialized.`
*   `[ SUCCESS ] QNN Context created on hardware.`
*   `[ RESULT ] QNN NATIVE WORKFLOW: HEALTHY`

---

### 1. Visibility (Is it working?)
The `inference_dinov3` provides "Readiness Verification". When it prints:
`[ SUCCESS ] Context created.`
`[ RESULT ] QNN Native Inference Readiness VERIFIED.`
It means the **entire hardware pipeline is open**. The model is loaded into the CPU/DSP memory, and the preprocessed image data has been mapped to the model's memory space.

### 2. Output Verification
The preprocessed image at `test/input.raw` is 602,112 bytes (224x224 RGB float32). 
To see the model's raw numerical output (embeddings), you can check the `test/output` directory after running the SDK's `qnn-net-run` tool.

---

## 💡 Troubleshooting: The "Unsupported Platform" Error
If you see `Error Code: 0x000003ee`, it means the bundled SDK v2.41 library is having a mismatch with the Ubuntu 24.04 kernel. 
**Fix**: Always use the system backend path `/usr/lib/libQnnCpu.so` as it is specifically tuned for this EVK's hardware.
