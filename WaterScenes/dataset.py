import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from radar_map_generate import RESOLUTION
from PIL import Image
import torchvision.transforms as transforms


# --------- Revised by songhee-cho: num_sample @ class WaterScenesDataset--------- #
class WaterScenesDataset(Dataset):
    def __init__(self, dataset_path='data', input_size=(RESOLUTION, RESOLUTION), num_classes=7, num_sample=10):
        self.dataset_path = dataset_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_sample = num_sample

        self.image_files = sorted(os.listdir(os.path.join(dataset_path, 'image')))[:self.num_sample]
        self.radar_files = sorted(os.listdir(os.path.join(dataset_path, 'radar')))[:self.num_sample]
        self.label_files = sorted(os.listdir(os.path.join(dataset_path, 'detection')))[:self.num_sample]

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def load_radar_data(self, radar_path):
        """CSV 파일에서 레이더 데이터를 REVP Map으로 변환"""
        radar_data = pd.read_csv(radar_path)
        radar_map = np.zeros((4, *self.input_size))  # (4, 640, 640)

        for _, row in radar_data.iterrows():
            u, v = int(row['u']), int(row['v'])
            if 0 <= u < self.input_size[0] and 0 <= v < self.input_size[1]:
                radar_map[:, v, u] = [row['range'], row['elevation'], row['doppler'], row['power']]

        return torch.tensor(radar_map, dtype=torch.float32)

    def load_labels(self, label_path):
        """YOLO 형식 바운딩 박스 로드"""
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    boxes.append(values)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        MAX_BOXES = 10
        # ✅ 이미지 로드
        img_path = os.path.join(self.dataset_path, 'image', self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        #print(f"image shape: {image.shape}") # Revised by songhee-cho

        # ✅ 레이더 데이터 로드
        radar_path = os.path.join(self.dataset_path, 'radar', self.radar_files[idx])
        radar_revp = self.load_radar_data(radar_path)
        #print(f"radar_revp shape: {radar_revp.shape}") # Revised by songhee-cho

        # ✅ 바운딩 박스 (YOLO 형식)
        label_path = os.path.join(self.dataset_path, 'detection', self.label_files[idx])
        labels = self.load_labels(label_path)

        # ✅ 이미지 + 레이더 Fusion (7채널 입력)
        fused_input = (torch.cat((image, radar_revp), dim=0)) # C*H*W


        return fused_input, labels
