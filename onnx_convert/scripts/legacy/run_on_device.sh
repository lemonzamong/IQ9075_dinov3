#!/bin/bash

cd /home/ubuntu/dinov3_deployment
echo '--- Starting Device Execution ---'
echo '--- Compiling ---'
echo 'Using existing processed weights...'
g++ -c -fPIC jni/QnnModel.cpp -I./include -I./include/QNN -I./jni
g++ -c -fPIC jni/QnnWrapperUtils.cpp -I./include -I./include/QNN -I./jni
g++ -c -fPIC jni/linux/QnnModelPal.cpp -I./include -I./include/QNN -I./jni
g++ -c -fPIC dinov3_qnn.cpp -I./include -I./include/QNN -I./jni
g++ -shared -fPIC -o bin/libdinov3.so dinov3_qnn.o QnnModel.o QnnWrapperUtils.o QnnModelPal.o @weights_objs.txt -I./include -I./include/QNN -I./jni
g++ -o bin/inference_dinov3 inference_dinov3.cpp -ldl -I./include -I./include/QNN -I./jni
export LD_LIBRARY_PATH=/home/ubuntu/dinov3_deployment/lib:$LD_LIBRARY_PATH
./bin/inference_dinov3 /home/ubuntu/dinov3_deployment/bin/libdinov3.so /home/ubuntu/dinov3_deployment/lib/libQnnCpu.so
echo '--- Verifying with qnn-net-run ---'
export LD_LIBRARY_PATH=/home/ubuntu/dinov3_deployment/lib:$LD_LIBRARY_PATH
./bin/qnn-net-run --backend /home/ubuntu/dinov3_deployment/lib/libQnnCpu.so --model /home/ubuntu/dinov3_deployment/bin/libdinov3.so --input_list /home/ubuntu/dinov3_deployment/test/input_list.txt --output_dir /home/ubuntu/dinov3_deployment/test/output
