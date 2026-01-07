# Native QNN (HTP) Implementation Plan for DINOv3

This plan transitions the DINOv3 deployment on Qualcomm IQ-9075 EVK from a hybrid ONNX Runtime/Native setup to a pure **Native QNN** implementation running on the **HTP (Hexagon)** backend, removing all ONNX Runtime dependencies as requested.

## User Review Required
> [!IMPORTANT]
> **HTP Backend Requirements**: Running on HTP requires the model to be quantized (usually INT8). The current `convert_on_host.sh` does not appear to perform quantization. I will update it to include quantization steps (using `qnn-onnx-converter` with calibration inputs), but this requires a representative calibration dataset (even 1-2 images).

> [!WARNING]
> **Custom Ops**: You mentioned `qnn-onnx-converter` does not support some DINOv3 operations. I will create the structure for a **QNN Custom Op Package** (`OpPackageGenerator`), but filling in the actual kernel implementation (C++/Hexagon DSP ASM) requires knowing exactly *which* operations are failing. I will add a step to identify these.

## Proposed Changes

### 1. Remove ONNX Runtime (ORT) Support
- **File**: `scripts/deploy.py`
    - Remove `--mode ort` logic.
    - Remove ORT-specific path usage.
    - Default to Native QNN mode.
- **File**: `scripts/inference_ort.py`
    - [DELETE] File.
- **File**: `README.md`
    - Update documentation to reflect Native-only workflow.

### 2. Enable HTP Backend & Quantization
- **File**: `native_qnn/convert_on_host.sh`
    - Update `qnn-onnx-converter` calls to include quantization parameters (or use `qnn-context-binary-generator` for HTP context).
    - Add logic to generate calibration data (raw files) from `assets/` or `test/`.
    - Change target to HTP-compatible graph.
- **File**: `scripts/deploy.py`
    - Update "Probe Device" logic to look for `libQnnHtp.so` and related DSP variants (e.g., `libQnnHtpV73.so`).
    - Update `qnn-net-run` command to use `--backend libQnnHtp.so`.

### 3. Custom Op Package Structure
- **Directory**: `native_qnn/custom_op/` [NEW]
    - Create a skeleton for custom op package generation using `qnn-op-package-generator` templates.
    - This allows you to plug in the unsupported op implementation later.

### 4. Inference & Verification (1280x720)
- **File**: `scripts/preprocess_input.py`
    - Update to support resizing input images to **1280x720** (or whatever the model expects - standard DINOv3 is often 224x224 or 518x518. **Clarification Needed**: Does the model *input* take 1280x720, or do you want to infer *on* a 1280x720 image (resized to model input)? Assuming resizing 1280x720 -> Model Input).
- **File**: `scripts/deploy_and_run.py` (or modify `deploy.py`)
    - Add parsing of `qnn-net-run` output to extract **inference time (ms)**.
    - Save timing logs.

## Verification Plan

### Automated Tests
1.  **Deployment Test**: Run `python scripts/deploy.py`.
    - Validates: Connection, File Transfer, On-device compilation, Execution.
    - Criteria: `qnn-net-run` completes with `exit 0`.
2.  **Performance Check**:
    - The script will output "Avg Inference Time: X ms".
    - Check if running on HTP (much faster than CPU).

### Manual Verification
1.  **Custom Op Check**: Run usage of `qnn-onnx-converter`. If it fails with "Unsupported Op", verify the Custom Op Package skeleton.
2.  **Image Output**: Copy `output/` from device to host and visualize.
