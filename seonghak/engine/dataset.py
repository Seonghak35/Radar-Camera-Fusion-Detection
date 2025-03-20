import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils import preprocess_input_radar, preprocess_input, cvtColor, resize_image

RESOLUTION = 160

# ✅ WaterScenes 데이터셋 클래스
class RadarCameraYoloDataset(Dataset):
    def __init__(self, data_root="WaterScenes/dataset/",
                 input_shape=(RESOLUTION, RESOLUTION), num_classes=7, transform=None):
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
        self.label_dir = os.path.join(data_root, "detection/yolo")

        self.input_shape = input_shape
        self.num_classes = num_classes
        # self.letterbox_image = False
        # self.transform = transform if transform else transforms.Compose([
        #     transforms.Resize(input_shape),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization for RGB image
        # ])
        self.resize_transform = transform if transform else transforms.Compose([
            transforms.Resize(input_shape),
        ])

        # 이미지 파일 리스트 가져오기
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 파일명 가져오기 (확장자 제외)
        file_name = os.path.splitext(self.image_files[idx])[0]

        # # 1️⃣ 이미지 불러오기 (RGB), referring to "Achelous"
        # image_path = os.path.join(self.image_dir, file_name + ".jpg")
        # image = Image.open(image_path)
        # image_shape = np.array(np.shape(image)[0:2])
        # image = cvtColor(image)
        # image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # image_data = np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1))  # (3, H, W)
        # image_data = torch.tensor(image_data, dtype=torch.float32)

        # 1️⃣ 이미지 불러오기 (RGB), referring to "Achelous"
        image_path = os.path.join(self.image_dir, file_name + ".jpg")
        image = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        # image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_data = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))  # (3, H, W)
        image_data = torch.tensor(image_data, dtype=torch.float32)
        image_data = self.resize_transform(image_data)

        ## previous version ##
        # image = Image.open(image_path).convert("RGB")
        # image = self.transform(image)  # (3, H, W)

        # 2️⃣ 레이더 REVP 데이터 불러오기 (.npz), referring to "Achelous"
        radar_path = os.path.join(self.radar_dir, file_name + ".npz")
        radar_data = np.load(radar_path)['arr_0']  # (4, H, W) → REVP 맵
        # radar_data = torch.from_numpy(preprocess_input_radar(radar_data)).type(torch.FloatTensor).unsqueeze(0)
        radar_data = preprocess_input_radar(radar_data)
        radar_data = torch.tensor(radar_data, dtype=torch.float32)

        ## previous version ##
        # radar_revp = torch.tensor(radar_data, dtype=torch.float32)

        # 3️⃣ 라벨 불러오기 (.txt)
        label_path = os.path.join(self.label_dir, file_name + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    labels.append(values)

        # YOLO 형식 라벨을 Tensor로 변환 (M, 5) → [class_id, x_center, y_center, width, height]
        labels = torch.tensor(labels, dtype=torch.float32) if len(labels) > 0 else torch.zeros((0, 5), dtype=torch.float32)

        # if len(labels) > 0:
        #     labels = torch.tensor(labels, dtype=torch.float32)
        # else:
        #     labels = torch.zeros((0, 5), dtype=torch.float32)  # 객체 없는 경우 빈 Tensor

        return image_data, radar_data, labels
    

# ✅ COCO Datasets Class formated in YOLO
class CoCoYoloDataset(Dataset):
    def __init__(self, data_root="../data/coco/",
                 input_shape=(320, 240), num_classes=80, transform=None):

        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "images")
        self.label_dir = os.path.join(data_root, "labels")

        self.input_shape = input_shape
        self.num_classes = num_classes
        # self.letterbox_image = True
        # self.transform = transform if transform else transforms.Compose([
        #     transforms.Resize(input_shape),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization for RGB image
        # ])
        self.resize_transform = transform if transform else transforms.Compose([
            transforms.Resize(input_shape),
        ])

        # 이미지 파일 리스트 가져오기
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 파일명 가져오기 (확장자 제외)
        file_name = os.path.splitext(self.image_files[idx])[0]

        # # 1️⃣ 이미지 불러오기 (RGB), referring to "Achelous"
        # image_path = os.path.join(self.image_dir, file_name + ".jpg")
        # image = Image.open(image_path)
        # image_shape = np.array(np.shape(image)[0:2])
        # image = cvtColor(image)
        # image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # image_data = np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1))  # (3, H, W)
        # image_data = torch.tensor(image_data, dtype=torch.float32)

        # 1️⃣ 이미지 불러오기 (RGB), referring to "Achelous"
        image_path = os.path.join(self.image_dir, file_name + ".jpg")
        image = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        # image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_data = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))  # (3, H, W)
        image_data = torch.tensor(image_data, dtype=torch.float32)
        image_data = self.resize_transform(image_data)

        ## previous version ##
        # image = Image.open(image_path).convert("RGB")
        # image = self.transform(image)  # (3, H, W)

        # 2️⃣ 라벨 불러오기 (.txt)
        label_path = os.path.join(self.label_dir, file_name + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    labels.append(values)

        # YOLO 형식 라벨을 Tensor로 변환 (M, 5) → [class_id, x_center, y_center, width, height]
        labels = torch.tensor(labels, dtype=torch.float32) if len(labels) > 0 else torch.zeros((0, 5), dtype=torch.float32)

        # if len(labels) > 0:
        #     labels = torch.tensor(labels, dtype=torch.float32)
        # else:
        #     labels = torch.zeros((0, 5), dtype=torch.float32)  # 객체 없는 경우 빈 Tensor

        return image_data, labels
