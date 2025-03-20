import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms, boxes

def preprocess_input_radar(data):
    _range = np.max(data) - np.min(data) # min-max norm
    data = (data - np.min(data)) / _range + 0.0000000000001 # avoid 0-value
    return data

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    # Bounding Box 변환: (x, y, w, h) -> (x_min, y_min, x_max, y_max)
    box_corner          = prediction.clone()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not conf_mask.any():  # Detection이 없는 경우 처리
            continue
        
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        # NMS 적용
        nms_scores = detections[:, 4] * detections[:, 5]  # Objectness * Class Confidence
        nms_out_index = torch.ops.torchvision.nms(detections[:, :4], nms_scores, nms_thres)

        output[i] = detections[nms_out_index]
        
        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    return output


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


def coco_collate_fn(batch):
    cameras = []
    labels = []

    for camera, label in batch:
        cameras.append(camera)
        labels.append(label)  # ✅ 리스트로 유지 (Tensor 변환 X)

    # ✅ 이미지 및 레이더 데이터를 스택
    cameras = torch.stack(cameras, dim=0)

    return cameras, labels  # ✅ `labels`은 리스트로 유지


def bbox_iou(box1, box2):
    """Compute IoU between two sets of bounding boxes in a batched way."""

    box1 = torch.tensor(box1, dtype=torch.float32)  # numpy → torch 변환
    box2 = torch.tensor(box2, dtype=torch.float32)  # numpy → torch 변환
    
    N = box1.shape[0]  # predicted objects
    M = box2.shape[0]  # real objects

    # (N, 1) → (N, M) 으로 확장 / (1, M) → (N, M) 으로 확장
    box1_x1 = (box1[:, 0] - box1[:, 2] / 2).unsqueeze(1).expand(N, M)
    box1_y1 = (box1[:, 1] - box1[:, 3] / 2).unsqueeze(1).expand(N, M)
    box1_x2 = (box1[:, 0] + box1[:, 2] / 2).unsqueeze(1).expand(N, M)
    box1_y2 = (box1[:, 1] + box1[:, 3] / 2).unsqueeze(1).expand(N, M)

    box2_x1 = (box2[:, 0] - box2[:, 2] / 2).unsqueeze(0).expand(N, M)
    box2_y1 = (box2[:, 1] - box2[:, 3] / 2).unsqueeze(0).expand(N, M)
    box2_x2 = (box2[:, 0] + box2[:, 2] / 2).unsqueeze(0).expand(N, M)
    box2_y2 = (box2[:, 1] + box2[:, 3] / 2).unsqueeze(0).expand(N, M)

    # 교차 영역 (Intersection)
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # 개별 박스 영역 (Union)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)  # (N, M) 형태의 IoU 행렬 반환

