# Project Architecture & Workflow Explanation
**Target Audience**: Graduate Student / Researcher  
**Goal**: Deploy Dino v3 (Vision Transformer) on Qualcomm IQ-9075 Edge Device.

## 1. The Big Picture
We are moving a heavy AI model from a "Training/Research" environment (your powerful PC/Server) to an "Inference/Edge" environment (the IQ-9075 kit).

**Why is this hard?**
-   **Hardware Differences**: Your PC uses NVIDIA GPUs (CUDA). The IQ-9075 uses a Qualcomm Hexagon NPU (QNN). They speak different languages.
-   **Software Compatibility**: PyTorch (used for training) is heavy and slow on edge devices. We need a lightweight, optimized runtime.

**The Solution Pipeline:**
```mermaid
graph LR
    A[PyTorch Model<br>(Hugging Face)] -->|Export| B(ONNX Format)
    B -->|Transfer| C[Edge Device<br>(IQ-9075)]
    C -->|QNN Runtime| D[Inference<br>Hardware Accelerated]
```

---

## 2. Key Concepts

### A. ONNX (Open Neural Network Exchange)
-   **What**: An open standard format for ML models. Think of it like a `.pdf` for AI. No matter if you wrote the model in PyTorch, TensorFlow, or JAX, you convert it to ONNX so other runtimes can read it.
-   **Why we use it**: Qualcomm's tools understand ONNX very well. It's the bridge between PyTorch and the NPU.

### B. QNN (Qualcomm AI Engine Direct)
-   **What**: Qualcomm's low-level software SDK. It allows code to talk directly to the Snapdragon hardware (CPU, GPU, and NPU/DSP).
-   **Execution Providers (EP)**: We use **ONNX Runtime** with the **QNN Execution Provider**. This is a plugin system.
    -   *Standard ONNX Runtime*: "I will run this math on the CPU." (Slow)
    -   *QNN Execution Provider*: "Wait, I see a Qualcomm NPU here. Let me offload these heavy matrix multiplications to the NPU." (Fast)

---

## 3. Directory Structure & File Roles

All your work is in `~/onnx_convert`. Here is what each file does:

### Host Side (Your PC)
1.  **`export_dinov3.py`**
    -   *Role*: The "Translator".
    -   *Logic*: Downloads Dino v3 from Hugging Face -> Traces the computation graph -> Saves it as `dinov3.onnx`.
    -   *Key Detail*: We export with `opset_version=17` to ensure compatibility with modern ONNX Runtimes.

2.  **`3_deploy_and_run.py`** (The "Master Script")
    -   *Role*: The "Automator".
    -   *Logic*:
        1.  SSH into the device (192.168.0.202).
        2.  Checks: "Do you already have the model file?" (If yes, skip upload. Saves time).
        3.  Sets up the environment (Make directory, install Python libs).
        4.  Executes the inference script remotely.
    -   *Why*: Manually copying files and running commands is prone to error. This script ensures a reproducible experiment every time.

### Device Side (IQ-9075)
1.  **`inference_qnn_onnx.py`**
    -   *Role*: The "Runner".
    -   *Logic*:
        1.  Loads `dinov3.onnx`.
        2.  **Crucial Step**: It attempts to load the QNN backend (`libQnnCpu.so` or `libQnnHtp.so`).
        3.  Runs a "Dummy Inference" (random noise image) just to prove the pipeline works.
    -   *Output*: Prints the shape of the output tensors. If you see shapes, the brain is working.

---

## 4. The Workflow You Just Completed

1.  **Export**: You ran `export_dinov3.py`. PyTorch model became `dinov3.onnx`.
2.  **Transfer**: `3_deploy_and_run.py` used `SCP` (Secure Copy Protocol) to send the 300MB+ model to the device.
3.  **Setup**: The script created a `venv` (Virtual Environment) on the device.
    -   *Why venv?* The device is a "Managed System" (Ubuntu). Installing libraries globally can break the OS. `venv` creates a sandbox for your project.
4.  **Inference**:
    -   The script found `/usr/lib/libQnnCpu.so` on the device.
    -   It told ONNX Runtime: "Use the QNN CPU backend library at this path."
    -   Result: Successful execution.

## 5. Next Steps for Research
 Now that the "Pipeline" works, your research can begin:
-   **Quantization**: Convert the model from Float32 to Int8. This is required to use the NPU (HTP backend) for max speed.
-   **Real Data**: Modify `inference_qnn_onnx.py` to load a real image (using `PIL` or `cv2`) instead of random noise.
-   **Benchmarking**: Measure how long the `session.run()` command takes. Compare CPU vs NPU performance.
