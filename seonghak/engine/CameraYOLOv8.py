import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pdb

from torch.utils.data import DataLoader, Dataset
from models import CSPBlock, ShuffleAttention, CameraYOLO
from dataset import RadarCameraYoloDataset
from utils import yolo_collate_fn, bbox_iou

# ✅ CUDA 강제 비활성화 (GPU 사용 금지)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ 강제 CPU 모드
device = torch.device("cpu")
print("⚠️ Running on CPU mode only")

# ✅ YOLO Loss Function
class YOLOLoss(nn.Module):
    def __init__(self, num_classes):
        super(YOLOLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        # preds: (B, N, num_classes + 5)
        # labels: List of Tensors [(M1, 5), (M2, 5), ...] -> [class_id, x_center, y_center, width, height]
        B, N, _ = preds.shape
        total_loss, total_bbox_loss, total_obj_loss, total_class_loss = 0, 0, 0, 0

        for i in range(B):
            if labels[i].numel() == 0:
                continue

            gt_classes = labels[i][:, 0].long()
            gt_bboxes = labels[i][:, 1:5]

            pred_bboxes = preds[i, :, :4]
            pred_obj = preds[i, :, 4]
            pred_cls = preds[i, :, 5:]

            # IoU 기반 GT 매칭
            iou_matrix = bbox_iou(pred_bboxes, gt_bboxes)  # (N, M)
            best_iou, best_pred_idx = iou_matrix.max(dim=0)  # (M,)

            matched_preds = pred_bboxes[best_pred_idx]  # (M, 4)
            matched_obj = pred_obj[best_pred_idx]  # (M,)
            matched_cls = pred_cls[best_pred_idx]  # (M, num_classes)

            bbox_loss = self.l1(matched_preds, gt_bboxes)  # (M, 4) vs (M, 4)
            obj_loss = self.bce(matched_obj, best_iou.detach())  # (M,)
            class_targets = torch.zeros_like(matched_cls)
            class_targets[range(gt_classes.shape[0]), gt_classes] = 1  # GT 클래스에 해당하는 위치에 1 할당
            class_loss = self.bce(matched_cls, class_targets)  # BCE 사용

            total_bbox_loss += bbox_loss
            total_obj_loss += obj_loss
            total_class_loss += class_loss
            total_loss += bbox_loss + obj_loss + class_loss

        return total_loss / B, total_bbox_loss / B, total_obj_loss / B, total_class_loss / B

# ✅ Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss, epoch_bbox_loss, epoch_obj_loss, epoch_class_loss = 0, 0, 0, 0
        total_samples = 0  # 전체 샘플 수
        
        for images, radar, labels in dataloader:
            batch_size = images.shape[0]  # 현재 배치 크기
            total_samples += batch_size

            images = images.to(device)
            labels = [label.to(device) for label in labels]

            optimizer.zero_grad()
            outputs = model(images)
            loss, bbox_loss, obj_loss, class_loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            epoch_bbox_loss += bbox_loss.item() * batch_size
            epoch_obj_loss += obj_loss.item() * batch_size
            epoch_class_loss += class_loss.item() * batch_size

        # 🔥 전체 샘플 수 기준으로 평균 Loss 계산
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / total_samples:.4f}, "
              f"Bbox Loss: {epoch_bbox_loss / total_samples:.4f}, "
              f"Obj Loss: {epoch_obj_loss / total_samples:.4f}, "
              f"Class Loss: {epoch_class_loss / total_samples:.4f}")
        
        # 🔥 10 Epoch마다 모델 저장 (output 폴더에 저장)
        if (epoch + 1) % 10 == 0:
            model.eval()
            model_path = f"output/trained_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved at epoch {epoch+1} in {model_path}")

# ✅ Model & Dataset Setup
device = torch.device("cpu")
num_classes = 7
split_ratio = 0.7

model = CameraYOLO(num_classes=num_classes).to(device)
dataset = RadarCameraYoloDataset(data_root="/workspaces/Radar-Camera-Fusion-Detection/WaterScenes/sample_dataset/")

# train_size = int(split_ratio * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

criterion = YOLOLoss(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 및 검증 루프
print("🚀 Training Started!")
train_model(model, train_loader, criterion, optimizer, num_epochs=50)
print("✅ Training Completed!")

# # ✅ 학습 완료 후 모델 저장
# model.eval()  # 평가 모드로 변경
# torch.save(model.state_dict(), "trained_model.pth")
# print("✅ Model saved as trained_model.pth")
