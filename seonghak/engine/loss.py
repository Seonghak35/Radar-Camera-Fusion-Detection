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
from utils import yolo_collate_fn, bbox_iou

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