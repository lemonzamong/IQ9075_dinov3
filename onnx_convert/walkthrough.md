# DINOv3 Native QNN Deployment (IQ-9075)

This guide details the verification of the DINOv3 model using **Native QNN (C++)** on the Hexagon Tensor Processor (HTP).

## Verification Results (2026-01-07)

### Executive Summary
The **Native QNN Deployment Pipeline** is fully functional and **HTP Hardware Acceleration** is verified.
- **HTP Verified**: Logs confirm execution on the DSP (`Finished Executing Graphs`).
- **Clean Deployment**: Legacy files (`onnx_convert`) removed from device; script uses `dinov3_deployment`.
- **Robust Transfer**: Deployment script uses MD5 checksums to ensure file integrity and skip unnecessary re-uploads.
- **Results Retrieved**: Inference outputs (`last_hidden_state.raw`, `pooler_output.raw`) are automatically downloaded to `output_results/`.

### Status Table
| Component | Status | Details |
| :--- | :--- | :--- |
| **Model Conversion** | ✅ **Pass** | ONNX -> QNN CPP/Bin (Quantized) successful. |
| **On-Device Compilation** | ✅ **Pass** | `libdinov3.so` compiles and links correctly on IQ-9075. |
| **HTP Inference** | ✅ **Pass** | `qnn-net-run` successfully executes graphs on DSP. |
| **Result Retrieval** | ✅ **Pass** | Outputs downloaded to host automatically. |

### Verification Logs (HTP)
```text
 <W> Initializing HtpProvider
 ...
 Composing Graphs
 Finalizing Graphs
 Executing Graphs
 Finished Executing Graphs
```

## How to Run

### 1. Prerequisites
```bash
source venv_qnn/bin/activate
pip install pillow paramiko scp glob2
```

### 2. Run Deployment (One-Click)
```bash
python3 scripts/deploy.py
```
*   **What it does**:
    1.  Cleans up legacy paths on device (`~/onnx_convert`).
    2.  Checks file integrity (MD5) and uploads assets/libs to `~/dinov3_deployment`.
    3.  Compiles model on-device.
    4.  Runs HTP inference.
    5.  Downloads results to `./output_results`.

### 3. Artifacts
**On Device (`~/dinov3_deployment/`)**:
- `bin/libdinov3.so`: Compiled model.
- `lib/hexagon/`: Hexagon DSP Skel libraries (Critical for HTP).
- `test/output/`: Raw inference results.

**On Host (`output_results/`)**:
- `output/Result_0/last_hidden_state.raw`
- `output/Result_0/pooler_output.raw`

## Running Inference on Custom Images

I have created a dedicated script (`scripts/inference.py`) to run inference on any new image using the deployed model.

### Usage
```bash
python3 scripts/inference.py <path_to_your_image.jpg> --output_dir <optional_output_dir>
```

### How it works
1.  **Preprocessing**: Resizes/Normalizes the image locally to `224x224` -> `temp_input.raw`.
2.  **Upload**: Sends raw input to device (`~/dinov3_deployment/test`).
3.  **Execution**: Runs `qnn-net-run` with **HTP Backend** on the device.
4.  **Timing**: Reports **Pure Inference Time** (excluding overhead) by parsing on-device logs.
5.  **Download**: Retrieves results (`last_hidden_state.raw`, `pooler_output.raw`) to your local output folder.

### Example
```bash
python3 scripts/inference.py my_cat.jpg
```
**Output**:
```text
[TIME] Preprocessing Time : 190.59 ms
[TIME] Inference Time     : 29.00 ms
Success! Results saved in inference_results
```
