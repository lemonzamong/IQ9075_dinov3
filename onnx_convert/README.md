# DINOv3 QNN Deployment on IQ-9075 EVK

This project deploys the **DINOv3** model on the **Qualcomm IQ-9075 Evaluation Kit** using the **Native QNN SDK** with **HTP (Hexagon Tensor Processor)** acceleration.

## Overview
- **Model**: DINOv3 (ViT-based)
- **Framework**: Native QNN (C++ API)
- **Backend**: HTP (Hardware Accelerated)
- **Input Resolution**: 224x224 (Images are automatically resized)
- **Key Features**:
    - **One-Click Deployment**: `deploy.py` handles uploading, on-device compilation, and verification.
    - **HTP Optimization**: Automatic uploading of correct Hexagon Skel libraries (`libQnnHtp*Skel.so`) to resolve firmware incompatibilities.
    - **Standalone Inference**: `inference.py` allows easy testing with custom images.

## Project Structure
```text
.
├── assets/                 # Model inputs (ONNX, calibration data)
├── native_qnn/
│   ├── convert_on_host.sh  # Script to convert ONNX -> QNN CPP/Bin
│   ├── src/                # C++ Source for on-device inference app
│   └── bin/                # Model weights (Large files)
├── scripts/
│   ├── deploy.py           # Main deployment & verification script
│   ├── inference.py        # Standalone inference script for custom images
│   └── preprocess_input.py # Image preprocessing utility
├── venv_qnn/               # Python virtual environment
└── output_results/         # Downloaded inference results (created automatically)
```

## Prerequisites

### Host (x86 Linux)
1.  **QNN SDK**: Installed at `/opt/qcom/aistack/qairt/<version>` (or similar).
2.  **Files**: Place `dinov3.onnx` in `assets/`.
3.  **Python Environment**:
    ```bash
    source venv_qnn/bin/activate
    pip install -r requirements.txt
    ```

## Workflow

### 0. Download Model (Host)
Downloads the ONNX model from Huggingface.
```bash
python3 scripts/download_model.py
```

### 1. Convert Model (Host)
Converts the ONNX model to QNN C++ source code and quantized binary weights.
```bash
cd native_qnn
./convert_on_host.sh
cd ..
```

### 2. Deploy & Verify (Host -> Device)
This script uploads all necessary assets, libraries (including Skel libs), compiles the model on the device, and runs a verification test.
```bash
python3 scripts/deploy.py
```
*   **Result**: Logs should show `Finished Executing Graphs` on HTP.
*   **Artifacts**: Results are downloaded to `output_results/`.

### 3. Run Inference (Custom Images)
To run the model on a new image:
```bash
python3 scripts/inference.py path/to/my_image.jpg
```
*   **Output**: Results (`last_hidden_state.raw`, etc.) are saved to `inference_results/`.

## Troubleshooting
- **CRC Mismatch / Unsupported SoC**: This usually means the device's DSP firmware is older than the SDK. `deploy.py` fixes this by uploading matching `*Skel.so` files from your SDK to `~/dinov3_deployment/lib/hexagon` and setting `ADSP_LIBRARY_PATH`.
- **Connection Failed**: Check the IP info in `scripts/deploy.py`.
