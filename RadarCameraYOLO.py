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

# CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
print("Running on CPU mode only")

# Define CSP block
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, expansion=0.5, downsample=False):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)

        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])     

        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y1 = self.bottlenecks(y1)
        y = torch.cat([y1, y2], dim=1)
        return self.final_conv(y)
    

# Define Shuffle attention
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=2):
        super(ShuffleAttention, self).__init__()
        self.groups = groups

        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W).transpose(1,2).reshape(B,C,H,W)
        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att
    
# WaterScenes dataset
class RadarCameraYoloDataset(Dataset):
    def __init__(self, data_root="/SSD/guest/teahyeon/Radar-Camera-Fusion-Detection/WaterScenes/data/",
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
    

class RadarCameraYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(RadarCameraYOLO, self).__init__()

        # Camera Feature Extractor (CSP)
        self.camera_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CSPBlock(64, 128, num_layers=3)
        )

        # Radar Feature Extractor
        self.radar_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.radar_deform_conv = DeformConv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.radar_bn = nn.BatchNorm2d(64)
        self.radar_attention = ShuffleAttention(64)

        # Adaptive fusion weight α
        self.alpha = nn.Parameter(torch.rand(1))

        # channel mathcing
        self.fusion_conv = nn.Sequential(
        nn.Conv2d(128 + 64, 128, kernel_size=3, stride=2, padding=1, bias=False),  
        nn.BatchNorm2d(128),
        nn.SiLU()
        )
        self.backbone_downsample = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)

        # # YOLOv8 Model load
        # self.channel_reduce = nn.Conv2d(192, 3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.yolo_model = YOLO("yolov8n")

        # YOLO Backbone (CSPDarknet)
        self.yolo_backbone = nn.Sequential(
            CSPBlock(128, 256, num_layers=3, downsample=True),
            CSPBlock(256, 512, num_layers=3, downsample=False),
            CSPBlock(512, 1024, num_layers=1, downsample=False)
        )

        # FPN Neck (Feature Pyramid Network)
        self.yolo_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # YOLO Decoupled Head
        self.yolo_head_cls = nn.Conv2d(256, num_classes, kernel_size=1)
        self.yolo_head_reg = nn.Conv2d(256, 4, kernel_size=1)
        self.yolo_head_obj = nn.Conv2d(256, 1, kernel_size=1) # Added by songhee-cho

    def forward(self, camera, radar):
        # Camera Feature Extraction
        F_camera = self.camera_stem(camera)

        # Radar Feature Extration
        radar_pooled = self.radar_pool(radar)

        stride_factor = self.radar_deform_conv.stride[0] if isinstance(self.radar_deform_conv.stride, tuple) else self.radar_deform_conv.stride
        offset_h = radar.size(2) // stride_factor
        offset_w = radar.size(3) // stride_factor
        offset = torch.zeros((radar.size(0), 18, offset_h, offset_w), device=radar.device)

        radar_feature = torch.relu(self.radar_deform_conv(radar_pooled, offset))
        radar_feature = self.radar_bn(radar_feature)
        radar_feature = self.radar_attention(radar_feature)

        # Adaptive Fusion
        fusion_feature = torch.cat([F_camera, self.alpha * radar_feature], dim=1)

        # Well-known YOLO --> for custom dataset, related code have to be added.
        # fusion_feature = self.channel_reduce(fusion_feature)
        # fusion_feature = fusion_feature / 255.0
        # yolo_output = self.yolo_model(fusion_feature)

        # Hakk codede YOLO
        fusion_feature = self.fusion_conv(fusion_feature)
        yolo_feature = self.yolo_backbone(fusion_feature)
        neck_feature = self.yolo_neck(yolo_feature)
        class_output = self.yolo_head_cls(neck_feature) # (B,num_classes,H,W)
        #bbox_output = self.yolo_head_reg(neck_feature) # (B, 4, H, W)
        obj_output = torch.sigmoid(self.yolo_head_obj(neck_feature)) # (B, 1, H, W)
        bbox_output = self.yolo_head_reg(neck_feature)  # (B, 4, H, W)
        bbox_output[:, :2] = torch.sigmoid(bbox_output[:, :2])  # center x, y → sigmoid
        bbox_output[:, 2:] = torch.exp(bbox_output[:, 2:])  # width, height → exp

        return class_output, bbox_output, obj_output

# ✅ Dynamic Collate Function (YOLO 바운딩 박스 개수 다름 문제 해결)
def yolo_collate_fn(batch):
    cameras = []
    radars = []
    labels = []

    for camera, radar, label in batch:
        cameras.append(camera)
        radars.append(radar)
        labels.append(label)  # ✅ 리스트로 유지 (Tensor 변환 X)

    # ✅ 이미지 및 레이더 데이터를 스택
    cameras = torch.stack(cameras, dim=0)
    radars = torch.stack(radars, dim=0)

    return cameras, radars, labels  # ✅ `labels`은 리스트로 유지


# ✅ IoU 계산 함수 수정
def compute_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2]
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    #print(f"Box1: {box1}, Box2: {box2}, IoU: {iou:.4f}")
    return iou

# ✅ 좌표 변환 함수 추가
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_center → x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_center → y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_center → x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_center → y2
    return y

# ✅ AP 계산 함수 수정
def compute_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

# ✅ mAP 계산 함수 수정
def compute_map(predictions, ground_truths, iou_threshold=0.5):
    aps = []
    for class_id in range(num_classes):
        gt_boxes = [gt[1:] for gt in ground_truths if int(gt[0]) == class_id]
        pred_boxes = [pred[1:] for pred in predictions if int(pred[0]) == class_id]

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # ✅ IoU 계산을 위해 좌표 변환 수행
        gt_boxes = xywh2xyxy(np.array(gt_boxes))
        pred_boxes = xywh2xyxy(np.array(pred_boxes))

        # ✅ Confidence 기준으로 정렬
        pred_boxes = sorted(pred_boxes, key=lambda x: x[-1], reverse=True)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()

        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou and j not in matched_gt:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        # ✅ Precision-Recall 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / len(gt_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # ✅ AP 계산 및 추가
        ap = compute_ap(precision, recall)
        aps.append(ap)

    return np.mean(aps) if aps else 0


class YOLOLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_box=1.0, lambda_obj=1.0):
        super(YOLOLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss()
        self.obj_loss = nn.BCEWithLogitsLoss()

        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj

    def forward(self, class_output, bbox_output, obj_output, target_classes_map, target_bboxes_map, target_obj_map):
        cls_loss = self.cls_loss(class_output.view(class_output.size(0), class_output.size(1), -1),
                                 target_classes_map.view(class_output.size(0), -1))
        
        box_loss = self.box_loss(bbox_output, target_bboxes_map)
        obj_loss = self.obj_loss(obj_output, target_obj_map)

        total_loss = self.lambda_cls * cls_loss + self.lambda_box * box_loss + self.lambda_obj * obj_loss
        
        return total_loss, cls_loss, box_loss, obj_loss

# ✅ 모델, 데이터 로더 설정
num_classes = 7
split_ratio = 0.7
model = RadarCameraYOLO(num_classes=num_classes).to(device)
# dataset = DummyRadarCameraYoloDataset(num_samples=1000)
# dataset = RadarCameraYoloDataset()
dataset = RadarCameraYoloDataset(data_root="WaterScenes/sample_dataset") # Revised by songhee-cho

train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

# ✅ 학습 및 검증 루프
num_epochs = 10
print("*************Training started!")

# ✅ 손실 함수 및 최적화 설정
loss_fn = YOLOLoss(lambda_cls=1.0, lambda_box=1.0, lambda_obj=1.0)  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    # ✅ 모델 학습 모드 설정
    model.train()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    
    for i, (camera, radar, labels) in enumerate(train_loader):
        camera, radar = camera.to(device), radar.to(device)

        # ✅ 모델 출력 (class_output, bbox_output, obj_output)
        class_output, bbox_output, obj_output = model(camera, radar)

        # ✅ Target 초기화
        target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
        target_bboxes_map = torch.zeros_like(bbox_output).to(device)
        target_obj_map = torch.zeros((obj_output.size(0), 1, obj_output.size(2), obj_output.size(3)), device=device)

        # ✅ Label → Target 변환
        for b, label in enumerate(labels):
            for obj in label:
                x_idx = int(obj[1] * class_output.size(2))
                y_idx = int(obj[2] * class_output.size(3))
                
                target_classes_map[b, y_idx, x_idx] = int(obj[0])
                target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]
                target_obj_map[b, 0, y_idx, x_idx] = 1.0  # Target existence → 1

        # ✅ 손실 계산
        total_loss, cls_loss, box_loss, obj_loss = loss_fn(class_output, bbox_output, obj_output,
                                                           target_classes_map, target_bboxes_map, target_obj_map)

        # ✅ 역전파 및 최적화
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ✅ 손실 값 집계
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()
        total_obj_loss += obj_loss.item()

        # ✅ 진행 상태 출력
        if i % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"Class Loss: {cls_loss.item():.4f}, BBox Loss: {box_loss.item():.4f}, "
                  f"Objectness Loss: {obj_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    # ✅ 에폭 별 평균 손실 출력
    print(f"✅ Epoch {epoch+1} Completed - "
          f"Class Loss: {total_cls_loss/len(train_loader):.4f}, "
          f"BBox Loss: {total_box_loss/len(train_loader):.4f}, "
          f"Objectness Loss: {total_obj_loss/len(train_loader):.4f}")

    # ✅ 모델 검증 모드 설정
    model.eval()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    map50, map75 = [], []

    with torch.no_grad():
        for i, (camera, radar, labels) in enumerate(val_loader):
            camera, radar = camera.to(device), radar.to(device)

            # ✅ 모델 출력
            class_output, bbox_output, obj_output = model(camera, radar)

            # ✅ Target 초기화
            target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
            target_bboxes_map = torch.zeros_like(bbox_output).to(device)
            target_obj_map = torch.zeros_like(obj_output).to(device)

            # ✅ Label → Target 변환
            for b, label in enumerate(labels):
                for obj in label:
                    x_idx = int(obj[1] * class_output.size(2))
                    y_idx = int(obj[2] * class_output.size(3))

                    target_classes_map[b, y_idx, x_idx] = int(obj[0])
                    target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]
                    target_obj_map[b,0, y_idx, x_idx] = 1.0

            # ✅ 손실 계산
            total_loss, cls_loss, box_loss, obj_loss = loss_fn(class_output, bbox_output, obj_output,
                                                               target_classes_map, target_bboxes_map, target_obj_map)

            # ✅ 손실 값 집계
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            total_obj_loss += obj_loss.item()

            # ✅ Confidence 계산
            class_prob = torch.softmax(class_output, dim=1)
            confidence, pred_classes = torch.max(class_prob, dim=1)

            predictions = []
            ground_truths = []

            for b, label in enumerate(labels):
                ground_truths.extend(label.cpu().numpy())

                pred_boxes = bbox_output[b].cpu().numpy().reshape(-1, 4)
                conf_scores = confidence[b].cpu().numpy().flatten()

                for j in range(pred_boxes.shape[0]):
                    if conf_scores[j] > 0.3:
                        predictions.append([
                            pred_classes[b].cpu().numpy().flatten()[j],
                            *pred_boxes[j],
                            conf_scores[j]
                        ])

            # ✅ mAP 계산 (IoU Threshold: 0.5, 0.75)
            map50.append(compute_map(predictions, ground_truths, iou_threshold=0.5))
            map75.append(compute_map(predictions, ground_truths, iou_threshold=0.75))

        # ✅ 검증 결과 출력
        print(f"✅ Validation - Epoch {epoch+1}, "
              f"Class Loss: {total_cls_loss/len(val_loader):.4f}, "
              f"BBox Loss: {total_box_loss/len(val_loader):.4f}, "
              f"Objectness Loss: {total_obj_loss/len(val_loader):.4f}")
        print(f"✅ Validation - mAP@50: {np.mean(map50):.4f}, mAP@75: {np.mean(map75):.4f}")

print("✅ Training Completed!")
save_path = "./trained_model.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")

