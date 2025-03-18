# ✅ IoU (Intersection over Union) 계산 함수
def compute_iou(box1, box2):
    """
    box1, box2: [x_center, y_center, width, height]
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ✅ Precision-Recall 곡선 기반 AP 계산
def compute_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


# ✅ mAP (Mean Average Precision) 계산 함수 (Confidence Score 추가)
def compute_map(predictions, ground_truths, iou_threshold=0.5, confidence_threshold=0.3):
    aps = []
    for class_id in range(num_classes):
        gt_boxes = [gt[1:] for gt in ground_truths if gt[0] == class_id]
        pred_boxes = [pred[1:] for pred in predictions if pred[0] == class_id and pred[-1] > confidence_threshold]

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        pred_boxes = sorted(pred_boxes, key=lambda x: x[-1], reverse=True)  # Confidence 기준 정렬

        tp, fp = np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes))
        matched_gt = set()

        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(pred, gt)
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
