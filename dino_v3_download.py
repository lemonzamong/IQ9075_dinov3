import torch
from transformers import AutoModel, AutoImageProcessor

# 1. 모델 및 프로세서 설정 (Hugging Face Hub에서 자동 다운로드)
model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
print(f"Loading model: {model_id}...")

# 모델 로드 (평가 모드로 설정)
model = AutoModel.from_pretrained(model_id)
model.eval()

# 2. 더미 입력 데이터 생성
# DINOv3는 기본적으로 224x224 크기의 이미지를 처리합니다.
# Shape: (Batch_Size, Channels, Height, Width) -> (1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. ONNX 변환 설정
output_file = "dinov3_vitb16.onnx"

print("Exporting to ONNX...")

# torch.onnx.export를 사용하여 변환
torch.onnx.export(
    model,
    dummy_input,
    output_file,
    export_params=True,        # 모델 파라미터(가중치) 포함
    opset_version=17,          # 최신 연산자 지원을 위해 14~17 권장 (QNN 호환성 고려)
    do_constant_folding=True,  # 상수 폴딩 최적화
    input_names=['pixel_values'],        # 입력 노드 이름 정의
    output_names=['last_hidden_state'],  # 출력 노드 이름 정의 (필요에 따라 pooler_output 추가 가능)
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},    # 배치 사이즈를 가변적으로 설정
        'last_hidden_state': {0: 'batch_size'}
    }
)

print(f"✅ Conversion complete! Saved as: {output_file}")