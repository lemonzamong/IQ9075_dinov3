#!/bin/bash

# Run this script on your IQ-9075 Device (Ubuntu)

# 0. Check for QNN SDK on Device
if [ -z "$QNN_SDK_ROOT" ]; then
    # Default path on device might be different, commonly /opt/qcom/aistack/qairt/<version>
    # Try to auto-detect
    DETECTED_SDK=$(find /opt/qcom/aistack/qairt -maxdepth 1 -name "2.*" | head -n 1)
    if [ -n "$DETECTED_SDK" ]; then
        export QNN_SDK_ROOT="$DETECTED_SDK"
        echo "Auto-detected QNN SDK at: $QNN_SDK_ROOT"
    else
        echo "Error: QNN_SDK_ROOT is not set and could not be auto-detected."
        echo "Please set it: export QNN_SDK_ROOT=/path/to/qnn/sdk"
        exit 1
    fi
fi

# Source env
if [ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ]; then
    source "$QNN_SDK_ROOT/bin/envsetup.sh"
fi

MODEL_LIB_PATH="./libs/aarch64-linux-clang/libdinov3.so"

if [ ! -f "$MODEL_LIB_PATH" ]; then
    echo "Error: $MODEL_LIB_PATH not found."
    echo "Did you run 1_convert_on_host.sh on your host and transfer the 'libs' folder here?"
    exit 1
fi

echo "--- Step 3: Building Inference Application ---"
mkdir -p build
cd build
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake failed."
    exit 1
fi

make
if [ $? -ne 0 ]; then
    echo "Error: Make failed."
    exit 1
fi
cd ..

echo "--- Step 4: Running Inference ---"
# We need to find the QNN CPU backend library to link/load
# It is usually in $QNN_SDK_ROOT/lib/aarch64-linux-clang/libQnnCpu.so
BACKEND_LIB="${QNN_SDK_ROOT}/lib/aarch64-linux-clang/libQnnCpu.so"
# Or HTP backend: libQnnHtp.so

echo "Running with backend: $BACKEND_LIB"
echo "Model Lib: $MODEL_LIB_PATH"

./build/inference_dinov3 "$MODEL_LIB_PATH" "$BACKEND_LIB"

echo "Done."
