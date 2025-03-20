import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from models import CSPBlock, ShuffleAttention, CameraYOLO
from dataset import RadarCameraYoloDataset, CoCoYoloDataset
from utils import yolo_collate_fn, bbox_iou, coco_collate_fn
from loss import YOLOLoss

# # ✅ CUDA 강제 비활성화 (GPU 사용 금지)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# # ✅ 강제 CPU 모드
# device = torch.device("cpu")
# print("⚠️ Running on CPU mode only")

# ✅ Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs, start_epoch=0, model_path=None):
    model.train()
    os.makedirs("output", exist_ok=True)  # 🔥 'output' 폴더 생성 (없으면 자동 생성)
    writer = SummaryWriter("logs")  # 🔥 TensorBoard writer 생성

    # 🔥 기존 모델이 있으면 불러오기
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded model from {model_path}, resuming training from epoch {start_epoch}")

    for epoch in range(num_epochs):
        epoch_loss, epoch_bbox_loss, epoch_obj_loss, epoch_class_loss = 0, 0, 0, 0
        total_samples = 0  # 전체 샘플 수
        
        for images, labels in tqdm(dataloader):
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
        avg_loss = epoch_loss / total_samples
        avg_bbox_loss = epoch_bbox_loss / total_samples
        avg_obj_loss = epoch_obj_loss / total_samples
        avg_class_loss = epoch_class_loss / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Bbox Loss: {avg_bbox_loss:.4f}, "
              f"Obj Loss: {avg_obj_loss:.4f}, "
              f"Class Loss: {avg_class_loss:.4f}")
        
        # 🔥 TensorBoard에 Loss 기록
        writer.add_scalar("Loss/Total", avg_loss, epoch + 1)
        writer.add_scalar("Loss/BBox", avg_bbox_loss, epoch + 1)
        writer.add_scalar("Loss/Objectness", avg_obj_loss, epoch + 1)
        writer.add_scalar("Loss/Class", avg_class_loss, epoch + 1)
        
        # 🔥 10 Epoch마다 모델 저장 (output 폴더에 저장)
        # if (epoch + 1) % 1 == 0:
        model.eval()
        model_path = f"output/trained_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved at epoch {epoch+1} in {model_path}")
    
    writer.close()  # 🔥 TensorBoard writer 종료


# ✅ Model & Dataset Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7
split_ratio = 0.2

model = CameraYOLO(num_classes=num_classes).to(device)
dataset = RadarCameraYoloDataset(data_root="../../WaterScenes/dataset/")
# dataset = CoCoYoloDataset(data_root="../data/coco/", input_shape=(160, 120))

train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=coco_collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

learning_rate = 5e-4
criterion = YOLOLoss(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ 학습 및 검증 루프
print("🚀 Training Started!")
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
# train_model(model, train_loader, criterion, optimizer, num_epochs=10, start_epoch=1, model_path='output/trained_model_epoch_1.pth')
print("✅ Training Completed!")

# # ✅ 학습 완료 후 모델 저장
# model.eval()  # 평가 모드로 변경
# torch.save(model.state_dict(), "trained_model.pth")
# print("✅ Model saved as trained_model.pth")
