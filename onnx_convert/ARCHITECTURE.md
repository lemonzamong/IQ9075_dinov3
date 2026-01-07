# System Architecture: Dino v3 on Qualcomm IQ-9075

This project implements two distinct deployment workflows for running the Dino v3 model on the Qualcomm IQ-9075 EVK. Both workflows utilize the Qualcomm AI Engine Direct (QNN) SDK but differ in their integration layer.

## 1. High-Level Overview

```mermaid
graph TD
    Host[Host Machine (x86_64)] -->|Scp/Ssh| Device[IQ-9075 Device (aarch64)]
    
    subgraph Host Step 1: Conversion
    HF[Hugging Face Model] -->|export_dinov3.py| ONNX[DinoV3 ONNX]
    ONNX -->|qnn-onnx-converter| QNN_Cpp[QNN C++ Model Code]
    ONNX -->|qnn-onnx-converter| QNN_Bin[QNN Binary Weights]
    end

    subgraph "Device Step 2: Deployment (Mode Selection)"
    QNN_Cpp -->|Native Mode| Native[Native C++ Application]
    ONNX -->|ORT Mode| ORT[ONNX Runtime Python App]
    end
```

## 2. Workflows

### A. ONNX Runtime (ORT) Mode
*   **Goal**: Rapid prototyping and ease of use.
*   **Mechanism**: Uses the Python `onnxruntime` library with the QNN Execution Provider (EP).
*   **Flow**:
    1.  Transfer `.onnx` model to device.
    2.  Run `scripts/inference_ort.py`.
    3.  ORT loads `libQnnHtp.so` or `libQnnCpu.so` dynamically to accelerate inference.

### B. Native QNN Mode (C++)
*   **Goal**: Maximum performance and minimal dependencies.
*   **Mechanism**: Compiles the model structure into a native shared library (`libdinov3.so`) and executes with the QNN API.
*   **Challenge**: The Host environment lacked a matching cross-compiler (`aarch64-linux-clang`) compatible with the SDK's build scripts.
*   **Solution: On-Device Compilation Strategy**
    To ensure robustness, the build process was moved to the device:
    1.  **Host**: Generate C++ model code (`dinov3_qnn.cpp`) and binary weights (`dinov3_qnn.bin`).
    2.  **Transfer**: Upload source code, weights, and crucial SDK headers/JNI sources to the IQ-9075.
    3.  **Device**:
        *   **Weight Processing**: The `.bin` file (tar archive) is extracted. `ld -r -b binary` converts raw weights into object files (`.o`).
        *   **Compilation**: `g++` compiles the model source and SDK wrappers (`QnnModel.cpp`, `QnnWrapperUtils.cpp`, `QnnModelPal.cpp`).
        *   **Linking**: All objects are linked into `libdinov3.so`.
        *   **Execution**: `inference_dinov3` (or `qnn-net-run`) loads the library and executes the graph.

## 3. Directory Structure

### Host (`onnx_convert/`)
| Directory | Description |
| :--- | :--- |
| `assets/` | ONNX models (`.onnx`), external data (`.data`), and QNN JSON configs. |
| `native_qnn/src/` | C++ source files (`dinov3_qnn.cpp` model source, `inference_dinov3.cpp` app). |
| `native_qnn/bin/` | Large binary weight files (`dinov3_qnn.bin`). |
| `scripts/` | Deployment (`deploy.py`) and utility scripts (preprocess, legacy). |
| `test/` | Test images (`test_image.jpg`) and inputs (`input.raw`). |

### Device (`~/dinov3_deployment/`)
Mirror of the host structure for execution.
| Directory | Description |
| :--- | :--- |
| `bin/` | Executables (`inference_dinov3`, `qnn-net-run`) and `libdinov3.so`. |
| `lib/` | QNN SDK libraries (`libQnnCpu.so`, etc.). |
| `assets/` | Model configs and metadata. |
| `test/` | Input data and output results. |

## 4. Key Components on Device
*   **`libQnnCpu.so`**: The backend runtime (found in bundled `lib/` or system `/usr/lib`).
*   **`libdinov3.so`**: The model-specific library we generate.
*   **`qnn-net-run`**: The Qualcomm standard tool for running network verification.
*   **`inference_dinov3`**: Custom C++ application acting as a standalone validator.
