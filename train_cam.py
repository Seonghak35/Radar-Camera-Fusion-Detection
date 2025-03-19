import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from CameraYOLO import CameraYOLO, YOLOLoss
from dataset import RadarCameraYoloDataset
from utils import compute_map, yolo_collate_fn

# Load model and data
def load_model_and_data(num_classes, split_ratio, data_root, device):
    model = CameraYOLO(num_classes=num_classes).to(device)
    dataset = RadarCameraYoloDataset(data_root=data_root)

    # Split dataset into train and val sets
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=yolo_collate_fn)

    return model, train_loader, val_loader

# Train function
def train(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    
    for i, (camera, _, labels) in enumerate(train_loader):
        camera = camera.to(device)

        # Forward pass
        class_output, bbox_output, obj_output = model(camera)

        # Target initialization
        target_classes_map = torch.zeros((class_output.size(0), class_output.size(2), class_output.size(3))).long().to(device)
        target_bboxes_map = torch.zeros_like(bbox_output).to(device)
        target_obj_map = torch.zeros_like(obj_output).to(device)

        # Convert label to target format
        for b, label in enumerate(labels):
            for obj in label:
                x_idx = int(obj[1] * class_output.size(2))
                y_idx = int(obj[2] * class_output.size(3))
                
                target_classes_map[b, y_idx, x_idx] = int(obj[0])
                target_bboxes_map[b, :, y_idx, x_idx] = obj[1:]
                target_obj_map[b, 0, y_idx, x_idx] = 1.0

        # Compute loss
        total_loss, cls_loss, box_loss, obj_loss = loss_fn(
            class_output, bbox_output, obj_output,
            target_classes_map, target_bboxes_map, target_obj_map
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()
        total_obj_loss += obj_loss.item()

        if i % 5 == 0:
            print(f"Epoch [{epoch+1}], Step [{i}/{len(train_loader)}], "
                  f"Class Loss: {cls_loss.item():.4f}, BBox Loss: {box_loss.item():.4f}, "
                  f"Objectness Loss: {obj_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    # Return averaged loss
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_box_loss = total_box_loss / len(train_loader)
    avg_obj_loss = total_obj_loss / len(train_loader)

    return avg_cls_loss, avg_box_loss, avg_obj_loss

# Validate function
def validate(model, val_loader, loss_fn, device, epoch):
    model.eval()
    total_cls_loss, total_box_loss, total_obj_loss = 0, 0, 0
    map50, map75 = [], []

    with torch.no_grad():
        for i, (camera, _, labels) in enumerate(val_loader):
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
            total_loss, cls_loss, box_loss, obj_loss = loss_fn(
                class_output, bbox_output, obj_output,
                target_classes_map, target_bboxes_map, target_obj_map
            )

            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            total_obj_loss += obj_loss.item()

            # Compute mAP
            predictions = []
            ground_truths = []
            class_prob = torch.softmax(class_output, dim=1)
            confidence, pred_classes = torch.max(class_prob, dim=1)

            for b, label in enumerate(labels):
                ground_truths.extend(label.cpu().numpy())
                pred_boxes = bbox_output[b].cpu().numpy().reshape(-1, 4)
                conf_scores = confidence[b].cpu().numpy().flatten()

                for j in range(pred_boxes.shape[0]):
                    if conf_scores[j] > 0.3:
                        predictions.append([pred_classes[b].cpu().numpy().flatten()[j], *pred_boxes[j], conf_scores[j]])

            map50.append(compute_map(predictions, ground_truths, iou_threshold=0.5))
            map75.append(compute_map(predictions, ground_truths, iou_threshold=0.75))

    # Return validation results
    return total_cls_loss / len(val_loader), total_box_loss / len(val_loader), total_obj_loss / len(val_loader), np.mean(map50), np.mean(map75)

# ✅ Save model function
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ✅ Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    split_ratio = 0.7
    data_root = "WaterScenes/sample_dataset"
    save_path = "./trained_model_cam.pth"
    num_epochs = 5

    model, train_loader, val_loader = load_model_and_data(num_classes, split_ratio, data_root, device)
    loss_fn = YOLOLoss(lambda_cls=1.0, lambda_box=1.0, lambda_obj=1.0)  
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(num_epochs):
        train(model, train_loader, loss_fn, optimizer, device, epoch)
        validate(model, val_loader, loss_fn, device, epoch)

    save_model(model, save_path)

# ✅ Execute the script
if __name__ == "__main__":
    main()
