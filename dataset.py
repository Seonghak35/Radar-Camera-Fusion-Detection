import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.ops import DeformConv2d
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
#from ultralytics import YOLO # for this, the ultralytics have to be installed.
import pdb
from WaterScenes.radar_map_generate import RESOLUTION # Revised by songhee-cho

# WaterScenes dataset
class RadarCameraYoloDataset(Dataset):
    def __init__(self, data_root="your_path",
                 input_shape=(RESOLUTION, RESOLUTION), num_classes=7, transform=None):
        """
        WaterScenes DataLoader

        :param data_root: data root dir
        :param input_shape: size of image and radar dataset
        :param num_classes: target class
        :param transform: for transform
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "image")
        self.radar_dir = os.path.join(data_root, "radar/REVP_map")
        self.label_dir = os.path.join(data_root, "detection")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization for RGB image
        ])

        # Load image file list
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load file name, except for extension
        file_name = os.path.splitext(self.image_files[idx])[0]

        # 1) Load RGB (.png)
        image_path = os.path.join(self.image_dir, file_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # (3, H, W)

        # 2) Load radar data (.npz)
        def preprocess_input_radar(data):
            _range = np.max(data) - np.min(data) # min-max norm
            data = (data - np.min(data)) / _range + 0.0000000000001 # avoid 0-value
            return data
            
        radar_path = os.path.join(self.radar_dir, file_name + ".npz")
        radar_data = np.load(radar_path)['arr_0']  # (4, H, W) → REVP 맵
        radar_data = preprocess_input_radar(radar_data)
        radar_revp = torch.tensor(radar_data, dtype=torch.float32)

        # 3) Load YOLO annotation (.txt)
        label_path = os.path.join(self.label_dir, file_name + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    labels.append(values)

        # YOLO Tensor (M, 5) → [class_id, x_center, y_center, width, height]
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)  # No object, 0-Tensor

        return image, radar_revp, labels