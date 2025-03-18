import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pdb

from torch.utils.data import DataLoader, Dataset
from engine.models import CSPBlock, ShuffleAttention, RadarCameraYOLO, yolo_collate_fn
from engine.dataset import RadarCameraYoloDataset

# ✅ CUDA 강제 비활성화 (GPU 사용 금지)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ 강제 CPU 모드
device = torch.device("cpu")
print("⚠️ Running on CPU mode only")


# ✅ 모델, 데이터 로더 설정
num_classes = 7
split_ratio = 0.7
model = RadarCameraYOLO(num_classes=num_classes).to(device)
dataset = RadarCameraYoloDataset(data_root="/workspaces/Radar-Camera-Fusion-Detection/WaterScenes/sample_dataset/")

train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

# ✅ 손실 함수 및 최적화 설정
cls_criterion = nn.CrossEntropyLoss() 
bbox_criterion = nn.SmoothL1Loss()
obj_criterion = nn.BCELoss()  # Objectness 손실
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 및 검증 루프
num_epochs = 1
print("🚀 Training started!")

def compute_loss(yolo_output, labels, device):
    """
    YOLOv8 스타일의 손실 함수.
    - yolo_output: (B, N, num_classes + 5)  # [x, y, w, h, obj, class_scores]
    - labels: (B, num_objects, 5)  # [class_id, x, y, w, h]
    """
    bbox_preds = yolo_output[:, :, :4]  # (B, N, 4)
    obj_preds = yolo_output[:, :, 4]  # (B, N)
    class_preds = yolo_output[:, :, 5:]  # (B, N, num_classes)

    B, N, num_classes = class_preds.shape

    # 🔹 Target 초기화
    target_bboxes = torch.zeros((B, N, 4), dtype=torch.float32, device=device)
    target_classes = torch.zeros((B, N), dtype=torch.long, device=device)
    target_objs = torch.zeros((B, N), dtype=torch.float32, device=device)  # Objectness label

    for b, label in enumerate(labels):
        if len(label) == 0:
            continue

        label = torch.as_tensor(label, dtype=torch.float32, device=device).clone().detach()  # ✅ 수정

        num_objects = label.shape[0]
        if num_objects > N:
            num_objects = N  # 객체 개수가 YOLO 출력 그리드 개수보다 많을 경우 제한

        # 🔹 객체가 있는 위치에 정답값 설정
        target_bboxes[b, :num_objects] = label[:num_objects, 1:]  # (num_objects, 4)
        target_classes[b, :num_objects] = label[:num_objects, 0].long()  # (num_objects,)
        target_objs[b, :num_objects] = 1.0  # Objectness 1

    # 🔥 Bounding Box Loss
    bbox_loss = bbox_criterion(bbox_preds, target_bboxes)

    # 🔥 Class Loss (객체가 있는 경우만 계산)
    cls_loss = cls_criterion(class_preds.view(-1, num_classes), target_classes.view(-1))

    # 🔥 Objectness Loss (BCE Loss)
    obj_loss = obj_criterion(obj_preds.view(-1), target_objs.view(-1))

    # ✅ 최종 Loss 계산
    loss = bbox_loss + cls_loss + obj_loss
    return loss, bbox_loss, cls_loss, obj_loss

# Training Loop
for epoch in range(50):
    for camera, radar, labels in train_loader:
        camera, radar = camera.to(device), radar.to(device)

        # 모델 Forward
        yolo_output = model(camera, radar)

        # 손실 계산
        loss, bbox_loss, cls_loss, obj_loss = compute_loss(yolo_output, labels, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Total Loss: {loss.item():.4f}, "
          f"BBox Loss: {bbox_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}, Obj Loss: {obj_loss.item():.4f}")


# for epoch in range(num_epochs):

#     # ✅ 학습 (Training)
#     model.train()
#     for i, (camera, radar, labels) in enumerate(train_loader):
#         camera, radar = camera.to(device), radar.to(device)

#         camera = camera.squeeze(1)  # (B, 1, 3, H, W) → (B, 3, H, W)
#         radar = radar.squeeze(1)  # (B, 1, 4, H, W) → (B, 4, H, W)

#         # class_output, bbox_output = model(camera, radar) # Model output
#         yolo_output = model(camera, radar)

#         # YOLO 출력 후처리: 바운딩 박스(boxes), 신뢰도(scores), 클래스(class_preds) 추출
#         bbox_preds = yolo_output[:, :, :4]  # (B, N, 4) - [x_center, y_center, w, h]
#         obj_preds = yolo_output[:, :, 4]  # (B, N) - Objectness Score
#         class_preds = yolo_output[:, :, 5:]  # (B, N, num_classes)
        
#         # 🔹 Target 초기화 (YOLO label과 동일한 형태)
#         target_bboxes = []
#         target_classes = []
#         target_objs = []

#         # 🔹 배치마다 Label을 YOLO 형식에 맞게 변환
#         for b, label in enumerate(labels):
#             if len(label) == 0:  # 객체가 없는 경우 처리
#                 continue

#             label = torch.tensor(label, device=device)  # (num_objects, 5) [class_id, x, y, w, h]
#             obj_target = torch.ones((label.shape[0],), device=device)  # Objectness Score = 1

#             # 저장 (배치 단위)
#             target_bboxes.append(label[:, 1:])  # 바운딩 박스 (x, y, w, h)
#             target_classes.append(label[:, 0].long())  # 클래스 ID
#             target_objs.append(obj_target)  # Objectness Target

#         # 🔹 리스트를 Tensor로 변환 (YOLO 출력과 동일한 크기로 맞추기)
#         if len(target_bboxes) > 0:
#             target_bboxes = torch.cat(target_bboxes, dim=0)  # (Total_objects, 4)
#             target_classes = torch.cat(target_classes, dim=0)  # (Total_objects,)
#             target_objs = torch.cat(target_objs, dim=0)  # (Total_objects,)
#         else:
#             target_bboxes = torch.zeros((0, 4), device=device)
#             target_classes = torch.zeros((0,), dtype=torch.long, device=device)
#             target_objs = torch.zeros((0,), device=device)

#         # 🔹 바운딩 박스 손실 계산 (Smooth L1 Loss)
#         bbox_loss = bbox_criterion(bbox_preds.view(-1, 4), target_bboxes)

#         # 🔹 클래스 손실 계산 (CrossEntropyLoss)
#         if target_classes.shape[0] > 0:
#             cls_loss = cls_criterion(class_preds.view(-1, num_classes), target_classes)
#         else:
#             cls_loss = torch.tensor(0.0, device=device)  # 객체가 없을 경우 손실 0

#         # 🔹 Objectness 손실 계산 (Binary Cross Entropy)
#         obj_loss = nn.BCELoss()(obj_preds.view(-1), target_objs)

#         # 🔥 전체 손실 계산
#         loss = bbox_loss + cls_loss + obj_loss

#         # 🔥 역전파 및 최적화
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # ✅ 로그 출력
#         if i % 5 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
#                   f"Cls Loss: {cls_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}, "
#                   f"Obj Loss: {obj_loss.item():.4f}, Total Loss: {loss.item():.4f}")

#     # ✅ Epoch 완료 메시지
#     print(f"✅ Epoch {epoch+1} completed.")

#     # # ✅ 검증 (Validation)
#     # model.eval()
#     # total_cls_loss, total_bbox_loss, map50, map75 = 0, 0, [], []

#     # with torch.no_grad():
#     #     for camera, radar, labels in val_loader:
#     #         camera, radar = camera.to(device), radar.to(device)

#     #         class_output, bbox_output = model(camera, radar)

#     #         target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
#     #         target_bboxes_map = torch.zeros_like(bbox_output).to(device)

#     #         for b, label in enumerate(labels):
#     #             for obj in label:
#     #                 x_idx = int(obj[1] * class_output.size(2))
#     #                 y_idx = int(obj[2] * class_output.size(3))
#     #                 target_classes_map[b, y_idx, x_idx] = int(obj[0])
#     #                 target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]

#     #         cls_loss = cls_criterion(class_output.view(class_output.size(0), class_output.size(1), -1),
#     #                                  target_classes_map.view(class_output.size(0), -1))
#     #         bbox_loss = bbox_criterion(bbox_output, target_bboxes_map)

#     #         total_cls_loss += cls_loss.item()
#     #         total_bbox_loss += bbox_loss.item()

#     # print(f"✅ Validation - Epoch {epoch+1}, Class Loss: {total_cls_loss/len(val_loader):.4f}, BBox Loss: {total_bbox_loss/len(val_loader):.4f}")
# print("✅ Training Completed!")

# # # ✅ 학습 완료 후 모델 저장
# # torch.save(model.state_dict(), "trained_model.pth")
# # print("✅ Model saved as trained_model.pth")
