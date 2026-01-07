cmake_minimum_required(VERSION 3.10)
project(DinoV3QnnInference)

set(CMAKE_CXX_STANDARD 17)

# Set QNN_SDK_ROOT environment variable or define it here
if(NOT DEFINED ENV{QNN_SDK_ROOT})
    message(WARNING "QNN_SDK_ROOT not set. Please set it to your QNN SDK installation path.")
    set(QNN_SDK_ROOT "/opt/qcom/qnn/sdk") # Default placeholder
else()
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
endif()

include_directories(${QNN_SDK_ROOT}/include/QNN)

add_executable(inference_dinov3 inference_dinov3.cpp)

# Link against dl for dynamic loading
target_link_libraries(inference_dinov3 dl)

# Note: You might need to link against QNN common libraries if using direct API calls
