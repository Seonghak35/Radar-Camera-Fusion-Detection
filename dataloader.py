import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pdb

# ✅ WaterScenes 데이터셋 클래스
class RadarCameraYoloDataset(Dataset):
    def __init__(self, data_root="/workspaces/Radar-Camera-Fusion-Detection/WaterScenes/data/",
                 input_shape=(160, 160), num_classes=7, transform=None):
        """
        WaterScenes DataLoader

        :param data_root: 데이터가 저장된 루트 디렉토리
        :param input_shape: 이미지 및 레이더 데이터의 크기
        :param num_classes: 객체 탐지 클래스 개수
        :param transform: 이미지 변환을 위한 torchvision.transforms
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "image")
        self.radar_dir = os.path.join(data_root, "radar/REVP_map")
        self.label_dir = os.path.join(data_root, "detection")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor()
        ])

        # 이미지 파일 리스트 가져오기
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 파일명 가져오기 (확장자 제외)
        file_name = os.path.splitext(self.image_files[idx])[0]

        # 1️⃣ 이미지 불러오기 (RGB)
        image_path = os.path.join(self.image_dir, file_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # (3, H, W)

        # 2️⃣ 레이더 REVP 데이터 불러오기 (.npz)
        radar_path = os.path.join(self.radar_dir, file_name + ".npz")
        radar_data = np.load(radar_path)['arr_0']  # (4, H, W) → REVP 맵
        radar_revp = torch.tensor(radar_data, dtype=torch.float32)

        # 3️⃣ 라벨 불러오기 (.txt)
        label_path = os.path.join(self.label_dir, file_name + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    labels.append(values)

        # YOLO 형식 라벨을 Tensor로 변환 (M, 5) → [class_id, x_center, y_center, width, height]
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)  # 객체 없는 경우 빈 Tensor

        return image, radar_revp, labels


# ✅ 데이터셋 로드 테스트
if __name__ == "__main__":
    dataset = RadarCameraYoloDataset()
    sample_image, sample_radar, sample_labels = dataset[0]

    print(f"Image Shape: {sample_image.shape}")  # (3, 160, 160)
    print(f"Radar Shape: {sample_radar.shape}")  # (4, 160, 160)
    print(f"Labels: {sample_labels}")  # (M, 5) → Bounding Box 정보
