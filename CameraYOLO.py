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
from dataset import RadarCameraYoloDataset

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
    
# Define CameraYOLO model
class CameraYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(CameraYOLO, self).__init__()

        # Camera Feature Extractor using CSPBlock
        self.camera_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # Initial downsampling
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CSPBlock(64, 128, num_layers=2, downsample=True) # First CSPBlock with downsampling
        )

        # YOLO Backbone using CSPDarknet structure
        self.yolo_backbone = nn.Sequential(
            CSPBlock(128, 256, num_layers=3, downsample=True),  # Downsampling and feature extraction
            CSPBlock(256, 512, num_layers=3, downsample=True),  # Further downsampling
            CSPBlock(512, 1024, num_layers=1, downsample=True) # Final downsampling for deep features
        )

        # FPN (Feature Pyramid Network) for multi-scale feature fusion
        self.yolo_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # YOLO Head (Classification, Regression, Objectness)
        self.yolo_head_cls = nn.Conv2d(256, num_classes, kernel_size=1)  # Predict class probabilities
        self.yolo_head_reg = nn.Conv2d(256, 4, kernel_size=1)  # Predict bounding box coordinates
        self.yolo_head_obj = nn.Conv2d(256, 1, kernel_size=1)  # Predict objectness score

    def forward(self, camera):
        # Pass through camera stem (feature extractor)
        camera_feature = self.camera_stem(camera)

        # Pass through YOLO backbone
        yolo_feature = self.yolo_backbone(camera_feature)

        # Pass through YOLO neck
        neck_feature = self.yolo_neck(yolo_feature)

        # Generate outputs from YOLO head
        class_output = torch.sigmoid(self.yolo_head_cls(neck_feature)) # Apply sigmoid to class probabilities
        bbox_output = self.yolo_head_reg(neck_feature)
        obj_output = torch.sigmoid(self.yolo_head_obj(neck_feature)) # Apply sigmoid to objectness score

        # Adjust bounding box predictions
        bbox_output[:, :2] = torch.sigmoid(bbox_output[:, :2])  # Center x, y -> sigmoid for normalization
        bbox_output[:, 2:] = torch.clamp(torch.exp(bbox_output[:, 2:]), max=10)  # Width, height -> exponential (with clamp to prevent overflow)

        return class_output, bbox_output, obj_output


# Dynamic Collate Function
def yolo_collate_fn(batch):
    cameras = []
    radars = []
    labels = []

    for camera, radar, label in batch:
        cameras.append(camera)
        radars.append(radar)
        labels.append(label)
    cameras = torch.stack(cameras, dim=0)
    radars = torch.stack(radars, dim=0)

    return cameras, radars, labels  # labels: list type


# IoU calculation
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

# xywh to xyxy
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_center → x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_center → y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_center → x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_center → y2
    return y

# AP
def compute_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

# mAP
def compute_map(predictions, ground_truths, iou_threshold=0.5):
    aps = []
    for class_id in range(num_classes):
        gt_boxes = [gt[1:] for gt in ground_truths if int(gt[0]) == class_id]
        pred_boxes = [pred[1:] for pred in predictions if int(pred[0]) == class_id]

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        gt_boxes = xywh2xyxy(np.array(gt_boxes))
        pred_boxes = xywh2xyxy(np.array(pred_boxes))
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

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / len(gt_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

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

# Load model, data
num_classes = 7
split_ratio = 0.7
model = CameraYOLO(num_classes=num_classes).to(device)
dataset = RadarCameraYoloDataset(data_root="WaterScenes/sample_dataset") # Revised by songhee-cho

train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

# Train, val loop
num_epochs = 100
print("*************Training started!*************")


# Loss function and optimization
loss_fn = YOLOLoss(lambda_cls=1.0, lambda_box=1.0, lambda_obj=1.0)  
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(num_epochs):
    # train mode on
    model.train()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    
    for i, (camera, _, labels) in enumerate(train_loader):
        camera = camera.to(device)

        # (class_output, bbox_output, obj_output)
        class_output, bbox_output, obj_output = model(camera)

        # Target initialization
        target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
        target_bboxes_map = torch.zeros_like(bbox_output).to(device)
        target_obj_map = torch.zeros((obj_output.size(0), 1, obj_output.size(2), obj_output.size(3)), device=device)

        # Label → Target 
        for b, label in enumerate(labels):
            for obj in label:
                x_idx = int(obj[1] * class_output.size(2))
                y_idx = int(obj[2] * class_output.size(3))
                
                target_classes_map[b, y_idx, x_idx] = int(obj[0])
                target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]
                #print(f"target_bboxes_map: {obj[1:]}")
                target_obj_map[b, 0, y_idx, x_idx] = 1.0  # Target existence → 1

        # Compute loss
        total_loss, cls_loss, box_loss, obj_loss = loss_fn(class_output, bbox_output, obj_output,
                                                           target_classes_map, target_bboxes_map, target_obj_map)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()
        total_obj_loss += obj_loss.item()

        # Print train loss
        if i % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"Class Loss: {cls_loss.item():.4f}, BBox Loss: {box_loss.item():.4f}, "
                  f"Objectness Loss: {obj_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    # Averaged loss
    print(f"Epoch {epoch+1} Completed - "
          f"Class Loss: {total_cls_loss/len(train_loader):.4f}, "
          f"BBox Loss: {total_box_loss/len(train_loader):.4f}, "
          f"Objectness Loss: {total_obj_loss/len(train_loader):.4f}")

    # val mode on
    model.eval()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    map50, map75 = [], []

    with torch.no_grad():
        for i, (camera,_, labels) in enumerate(val_loader):
            camera = camera.to(device)
            class_output, bbox_output, obj_output = model(camera)
            target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
            target_bboxes_map = torch.zeros_like(bbox_output).to(device)
            target_obj_map = torch.zeros_like(obj_output).to(device)

            for b, label in enumerate(labels):
                for obj in label:
                    x_idx = int(obj[1] * class_output.size(2))
                    y_idx = int(obj[2] * class_output.size(3))

                    target_classes_map[b, y_idx, x_idx] = int(obj[0])
                    target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]
                    target_obj_map[b,0, y_idx, x_idx] = 1.0

            # Compute loss
            total_loss, cls_loss, box_loss, obj_loss = loss_fn(class_output, bbox_output, obj_output,
                                                               target_classes_map, target_bboxes_map, target_obj_map)

            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            total_obj_loss += obj_loss.item()

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

            # Compute mAP (IoU Threshold: 0.5, 0.75)
            map50.append(compute_map(predictions, ground_truths, iou_threshold=0.5))
            map75.append(compute_map(predictions, ground_truths, iou_threshold=0.75))

        # Print validation loss
        print(f"Validation - Epoch {epoch+1}, "
              f"Class Loss: {total_cls_loss/len(val_loader):.4f}, "
              f"BBox Loss: {total_box_loss/len(val_loader):.4f}, "
              f"Objectness Loss: {total_obj_loss/len(val_loader):.4f}")
        print(f"Validation - mAP@50: {np.mean(map50):.4f}, mAP@75: {np.mean(map75):.4f}")

print("Training Completed!")
save_path = "./trained_model_cam.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")