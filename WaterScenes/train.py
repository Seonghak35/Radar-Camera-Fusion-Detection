import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WaterScenesDataset

# ✅ CPU 강제 실행
device = torch.device("cpu")

# ✅ 데이터 로드 (Fusion 데이터셋 사용)
dataset = WaterScenesDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# ✅ YOLO 모델 정의 (7채널 입력 지원)
class YOLOv8Fusion(nn.Module):
    def __init__(self, num_classes=7):
        super(YOLOv8Fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes + 4, kernel_size=1, stride=1, padding=0)  # 클래스 + 바운딩 박스 (x, y, w, h)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ✅ 모델 및 손실 함수 설정
model = YOLOv8Fusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ IoU 계산 함수
def compute_iou(pred_box, true_box):
    """
    IoU (Intersection over Union) 계산 함수
    """
    # (x, y, w, h) 형식
    x1_pred, y1_pred, w_pred, h_pred = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
    x1_true, y1_true, w_true, h_true = true_box[:, 0], true_box[:, 1], true_box[:, 2], true_box[:, 3]

    # 좌표 계산
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    x2_true, y2_true = x1_true + w_true, y1_true + h_true

    # 교차 영역 (Intersection) 계산
    x1_inter = torch.max(x1_pred, x1_true)
    y1_inter = torch.max(y1_pred, y1_true)
    x2_inter = torch.min(x2_pred, x2_true)
    y2_inter = torch.min(y2_pred, y2_true)

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

    # 개별 영역 (Union) 계산
    pred_area = w_pred * h_pred
    true_area = w_true * h_true
    union_area = pred_area + true_area - inter_area

    # IoU 계산
    iou = inter_area / (union_area + 1e-6)  # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    return iou.mean()

# ✅ YOLO Loss 정의 (IoU + Classification Loss)
def yolo_loss(outputs, labels):
    loss = 0
    batch_size, num_features, H, W = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).contiguous()  # (batch_size, H, W, num_features)

    for i in range(len(labels)):  # 각 이미지에 대해
        if len(labels[i]) == 0:  # 바운딩 박스가 없는 경우
            continue
        
        # ✅ YOLO Grid Cell 기반 바운딩 박스 변환
        pred_boxes = outputs[i].view(-1, num_features)  # (H*W, num_features)
        true_boxes = labels[i]  # (M, 5) (class, x, y, w, h)

        # ✅ IoU 기반 Loss 계산
        iou_loss = compute_iou(pred_boxes[:, -4:], true_boxes[:, 1:])

        # ✅ 클래스 예측 손실 계산
        class_pred = pred_boxes[:, :-4]  # 클래스 예측 부분
        class_target = true_boxes[:, 0].long()  # 클래스 ID
        class_loss = nn.CrossEntropyLoss()(class_pred, class_target)

        loss += iou_loss + class_loss

    return loss / len(labels)

# ✅ 학습 루프
for epoch in range(5):
    for fused_input, labels in dataloader:
        fused_input = fused_input.to(device)

        # ✅ Forward Pass
        outputs = model(fused_input)

        # ✅ YOLO Loss 적용
        loss = yolo_loss(outputs, labels)

        # ✅ Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

print("✅ Training Completed!")
