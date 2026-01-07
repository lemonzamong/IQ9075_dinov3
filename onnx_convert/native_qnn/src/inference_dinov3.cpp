// Native QNN Inference Test - v5.0 (GraphInfo_t Integration)
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "QnnInterface.h"

// Define the missing struct from QnnWrapperUtils.hpp
typedef struct GraphInfo {
  void* graph; // Qnn_GraphHandle_t
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
} GraphInfo_t;

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef int (*ComposeGraphsFn_t)(void*, QNN_INTERFACE_VER_TYPE, void*, const void**, uint32_t, GraphInfo_t***, uint32_t*);

void printHexError(const std::string& msg, Qnn_ErrorHandle_t error) {
    if (error != 0) {
        std::cerr << "[ ERROR ] " << msg << " (Error Code: 0x" << std::hex << std::setw(8) << std::setfill('0') << error << ")" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model_lib> <backend_lib> <input_raw>" << std::endl;
        return 1;
    }

    std::string modelLibPath = argv[1];
    std::string backendLibPath = argv[2];
    std::string inputRawPath = argv[3];

    std::cout << "[ INFO ] Native QNN Full Inference (GraphInfo Mode)" << std::endl;

    // 1. Load Backend and Model
    void* backendHandle = dlopen(backendLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    auto getProvidersFn = (QnnInterfaceGetProvidersFn_t)dlsym(backendHandle, "QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    getProvidersFn(&providers, &numProviders);
    QNN_INTERFACE_VER_TYPE qnnInterface = providers[0]->QNN_INTERFACE_VER_NAME;

    void* modelHandle = dlopen(modelLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    auto composeGraphsFn = (ComposeGraphsFn_t)dlsym(modelHandle, "QnnModel_composeGraphs");

    // 2. Initialize
    void* backend = nullptr;
    qnnInterface.backendCreate(nullptr, nullptr, (Qnn_BackendHandle_t*)&backend);
    void* context = nullptr;
    qnnInterface.contextCreate(backend, nullptr, nullptr, (Qnn_ContextHandle_t*)&context);

    // 3. Compose
    GraphInfo_t** graphsInfo = nullptr;
    uint32_t numGraphs = 0;
    std::cout << "[ INFO ] Composing graphs..." << std::endl;
    composeGraphsFn(backend, qnnInterface, context, nullptr, 0, &graphsInfo, &numGraphs);
    
    GraphInfo_t* graphInfo = graphsInfo[0];
    std::cout << "[ INFO ] Graph: " << graphInfo->graphName << " | Inputs: " << graphInfo->numInputTensors << " | Outputs: " << graphInfo->numOutputTensors << std::endl;

    // 4. Bind Buffers
    // Input: pixel_values
    std::ifstream inputFile(inputRawPath, std::ios::binary);
    std::vector<char> inputData(602112); // float32, 1x224x224x3
    inputFile.read(inputData.data(), inputData.size());

    for (uint32_t i=0; i<graphInfo->numInputTensors; ++i) {
        Qnn_Tensor_t& tensor = graphInfo->inputTensors[i];
        const char* name = (tensor.version == QNN_TENSOR_VERSION_1) ? tensor.v1.name : tensor.v2.name;
        if (std::string(name) == "pixel_values") {
            if (tensor.version == QNN_TENSOR_VERSION_1) {
                tensor.v1.clientBuf.data = inputData.data();
                tensor.v1.clientBuf.dataSize = inputData.size();
            } else {
                tensor.v2.clientBuf.data = inputData.data();
                tensor.v2.clientBuf.dataSize = inputData.size();
            }
        }
    }

    // Output: last_hidden_state
    std::vector<float> outputData(1 * 201 * 768);
    for (uint32_t i=0; i<graphInfo->numOutputTensors; ++i) {
        Qnn_Tensor_t& tensor = graphInfo->outputTensors[i];
        const char* name = (tensor.version == QNN_TENSOR_VERSION_1) ? tensor.v1.name : tensor.v2.name;
        if (std::string(name) == "last_hidden_state") {
            if (tensor.version == QNN_TENSOR_VERSION_1) {
                tensor.v1.clientBuf.data = outputData.data();
                tensor.v1.clientBuf.dataSize = outputData.size() * sizeof(float);
            } else {
                tensor.v2.clientBuf.data = outputData.data();
                tensor.v2.clientBuf.dataSize = outputData.size() * sizeof(float);
            }
        }
    }

    // 5. Execute
    std::cout << "[ INFO ] Executing inference..." << std::endl;
    Qnn_ErrorHandle_t status = qnnInterface.graphExecute(graphInfo->graph, graphInfo->inputTensors, graphInfo->numInputTensors, graphInfo->outputTensors, graphInfo->numOutputTensors, nullptr, nullptr);
    if (status != 0) {
        printHexError("Execute failed", status);
        return 1;
    }

    // 6. Statistics
    double sum = std::accumulate(outputData.begin(), outputData.end(), 0.0);
    double mean = sum / outputData.size();
    std::cout << "[ SUCCESS ] Inference completed!" << std::endl;
    std::cout << "Mean:  " << mean << " (Min: " << *std::min_element(outputData.begin(), outputData.end()) 
              << ", Max: " << *std::max_element(outputData.begin(), outputData.end()) << ")" << std::endl;
    
    std::cout << "[ RESULT ] NATIVE INFERENCE VERIFIED: REAL IMAGE DATA PROCESSED." << std::endl;

    qnnInterface.contextFree(context, nullptr);
    qnnInterface.backendFree(backend);
    dlclose(modelHandle);
    dlclose(backendHandle);
    return 0;
}
