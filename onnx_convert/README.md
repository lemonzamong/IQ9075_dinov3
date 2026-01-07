# Dino v3 on Qualcomm IQ-9075 (QNN)

This repository contains tools to export the Dino v3 model to ONNX and deploy it to the IQ-9075 Evaluation Kit for inference.

## Prerequisites
- **Host**: Python 3.8+
- **Device (IQ-9075)**: Connected to network, running Ubuntu (QNN SDK/Runtime recommended).

## Setup
1. Install Host Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 1. Export Model
Download `facebook/dinov3-vitb16-pretrain-lvd1689m` and export to ONNX.
```bash
python export_dinov3.py
```
*Output*: `dinov3.onnx`, `dinov3.onnx.data`

### 2. Verify Export (Optional)
Check if the exported ONNX model runs locally.
```bash
python verify_onnx.py
```

### 3. Deploy and Run on IQ-9075
Automated script to transfer files to the device (`192.168.0.202`) and run inference.
```bash
python 3_deploy_and_run.py
```

**What this does:**
1. Connects via SSH (`ubuntu/qualcomm`).
2. Uploads model and scripts.
3. Sets up a Python virtual environment on the device (to avoid system conflicts).
4. Installs `onnxruntime` and `numpy` on the device.
5. Runs `inference_qnn_onnx.py`.

---
## Advanced: C++ QNN Native Workflow
If you wish to use the native QNN C++ API (requires QNN SDK Host Tools installed locally):

1. **Convert to QNN Library (Host)**:
   ```bash
   chmod +x 1_convert_on_host.sh
   ./1_convert_on_host.sh
   ```
   *Requires `qnn-onnx-converter` in PATH.*

2. **Deploy (Device)**:
   The `3_deploy_and_run.py` script will automatically detect if you have generated `libs/` from step 1 and upload them. It will then attempt to compile and run the C++ application (`inference_dinov3.cpp`).

## Manual Execution (On Device)
If you are already SSH'd into the device (`192.168.0.202`) and typically see `~/onnx_convert`:

1. **Go to directory**:
   ```bash
   cd ~/onnx_convert
   ```

2. **Activate Environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Run Inference**:
   ```bash
   python inference_qnn_onnx.py --model dinov3.onnx --backend qnn_cpu --qnn_lib /usr/lib/libQnnCpu.so
   ```

**Verification:**
If you see output shapes like `(1, 201, 768)`, it is working correctly!
