import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("yolov8n")

# 더미 데이터 (640x640 입력)
dummy_input = torch.randn(1, 3, 640, 640)

# Backbone의 모든 Feature Map 확인
with torch.no_grad():
    x = dummy_input
    feature_maps = []  # ✅ 저장할 Feature Map 리스트

    for i, layer in enumerate(model.model.model[:-1]):  # ✅ 마지막 Detection Head 제외
        try:
            if isinstance(x, tuple):  # ✅ x가 튜플이면 Concat이 필요한 경우임
                print(f"⚠️ Concat 연산이 필요한 Layer {i}에서 중단.")
                break  # ✅ Concat 연산이 필요한 Layer 전까지만 출력하고 종료
            
            x = layer(x)
            feature_maps.append(x)
            print(f"Feature {i}: {x.shape}")  # ✅ 모든 Feature Map의 크기 출력
        except TypeError as e:
            print(f"🔥 TypeError at layer {i}: {e}")  # ❌ 에러 발생 위치 확인
            break  # ❌ 에러 발생 시 중단
