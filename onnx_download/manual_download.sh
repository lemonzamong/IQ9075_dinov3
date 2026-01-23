#!/bin/bash
MODEL_ID="facebook/dinov3-vit7b16-pretrain-lvd1689m"
TOKEN="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"
OUTPUT_DIR="dinov3_vit7b_pth"

mkdir -p $OUTPUT_DIR

echo "Removing potential partial files..."
rm -f "$OUTPUT_DIR/model-00005-of-00006.safetensors"
rm -f "$OUTPUT_DIR/model-00006-of-00006.safetensors"

echo "Downloading shard 5..."
wget --header="Authorization: Bearer $TOKEN" \
     https://huggingface.co/$MODEL_ID/resolve/main/model-00005-of-00006.safetensors \
     -O "$OUTPUT_DIR/model-00005-of-00006.safetensors"

echo "Downloading shard 6..."
wget --header="Authorization: Bearer $TOKEN" \
     https://huggingface.co/$MODEL_ID/resolve/main/model-00006-of-00006.safetensors \
     -O "$OUTPUT_DIR/model-00006-of-00006.safetensors"
