# Dino v3 QNN Deployment on IQ-9075

This project deploys the Dino v3 Vision Transformer model to the Qualcomm IQ-9075 Evaluation Kit using two methods:
1.  **Native QNN (C++)**: Compiled directly against the QNN SDK for maximum efficiency.
2.  **ONNX Runtime (Python)**: Uses the QNN Execution Provider for ease of integration.

## Prerequisites

*   **Host**: Linux (x86_64) with QNN SDK installed.
*   **Device**: IQ-9075 (aarch64) with QNN runtime libraries (`libQnnCpu.so`) in `/usr/lib`.
*   **Network**: Passwordless SSH access recommended (Script uses `ubuntu:qualcomm` by default).

## Installation

1.  **Setup Host Environment**:
    ```bash
    # Create venv with specific versions for QNN tools (Numpy < 2.0)
    python3.10 -m venv venv_qnn
    source venv_qnn/bin/activate
    pip install -r requirements.txt
    ```

2.  **Place Test Image (Optional)**:
    Place a `test_image.jpg` in this directory to automatically verify inference.

## Usage

Use the master deployment script `3_deploy_and_run.py` to handle everything (conversion, upload, compilation, execution).

Deploy and run using the Python script in `scripts/`.

### 1. Native QNN Mode (Verified)
Compiles C++ inference app on device and runs it.
```bash
python scripts/deploy.py --mode native
```
-   **Features**:
    -   Smart Reuse: Skips uploading large files (`.bin`, SDK libs) if they exist on device.
    -   On-Device Compilation: Automatically compiles model and wrappers.
    -   Verification: Runs `inference_dinov3` custom app to validate model loading.
    -   Inputs: Automatically preprocesses `test/test_image.jpg`.

### 2. ONNX Runtime Mode (Verified)
Runs generic ONNX inference via Python.
```bash
python scripts/deploy.py --mode ort
```

## Directory Structure (Host)
-   `scripts/`: Deployment (`deploy.py`) and utility (`preprocess_input.py`) scripts.
-   `native_qnn/src`: C++ source (`dinov3_qnn.cpp`, `inference_dinov3.cpp`).
-   `native_qnn/bin`: Large binary weights (`dinov3_qnn.bin`).
-   `assets/`: ONNX models and JSON configs.
-   `test/`: Test images.

## Directory Structure (Device)
Deployed to `~/dinov3_deployment`.
The script will automatically:
1.  Preprocess it to `input.raw` (1x3x224x224, normalized).
2.  Run inference on the device.
3.  Save outputs to `output/` directory on the device.

## Troubleshooting
*   **Connection Timed Out**: Ensure the device IP (`192.168.0.202`) is reachable.
*   **QNN_SDK_ROOT Error**: Ensure `export QNN_SDK_ROOT=/path/to/sdk` is set on the host.
