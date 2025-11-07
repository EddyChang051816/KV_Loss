import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou, nms
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v3_large
import time
import numpy as np
from torch.nn import KLDivLoss
import torch.nn.functional as F

# --- Configuration (please update paths before running) ---
TRAIN_IMG_DIR = r"./train_img"  # <-- update
TRAIN_XML_DIR = r"./train_xml"  # <-- update
TEST_IMG_DIR = r"./val_img"  # <-- update
TEST_XML_DIR = r"./val_xml"  # <-- update
BATCH_SIZE = 2
LEARNING_RATE = 1e-4


# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 標籤對應表
LABEL_MAP = {
    '18': 1, '17': 2, '16': 3, '15': 4, '14': 5, '13': 6, '12': 7, '11': 8,
    '21': 9, '22': 10, '23': 11, '24': 12, '25': 13, '26': 14, '27': 15, '28': 16,
    '48': 17, '47': 18, '46': 19, '45': 20, '44': 21, '43': 22, '42': 23, '41': 24,
    '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32
}

# 解析 VOC XML
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in LABEL_MAP:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(LABEL_MAP[label])

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

# 自定義 Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, target_size=(800, 800)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(target_size),   # ✅ 統一尺寸
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.xml').replace('.png', '.xml').replace('.jpeg', '.xml'))

        img = Image.open(img_path).convert("RGB")
        boxes, labels = parse_voc_xml(label_path)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        img = self.transform(img)  # ✅ 使用 transform
        return img, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, data_loader, device, nms_iou_threshold=0.5, show_confusion_matrix=False):
    model.eval()
    mAP_metric = MeanAveragePrecision(iou_thresholds=[0.5 + i * 0.05 for i in range(10)])
    all_preds = []
    all_targets = []


    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i, output in enumerate(outputs):
                keep_indices = nms(output['boxes'], output['scores'], nms_iou_threshold)
                pred_dict = {
                    'boxes': output['boxes'][keep_indices].detach().cpu(),
                    'scores': output['scores'][keep_indices].detach().cpu(),
                    'labels': output['labels'][keep_indices].detach().cpu()
                }
                target_dict = {
                    'boxes': targets[i]['boxes'].detach().cpu(),
                    'labels': targets[i]['labels'].detach().cpu()
                }

                if len(pred_dict['labels']) > 0 and len(target_dict['labels']) > 0:
                    mAP_metric.update([pred_dict], [target_dict])
                    iou_matrix = torchvision.ops.box_iou(pred_dict['boxes'], target_dict['boxes'])
                    for pred_idx, target_idx in zip(*torch.where(iou_matrix > 0.5)):
                        all_preds.append(pred_dict['labels'][pred_idx].item())
                        all_targets.append(target_dict['labels'][target_idx].item())

    metrics = mAP_metric.compute()
    precision, recall, f1, accuracy = 0, 0, 0, 0

    if len(all_targets) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

        if show_confusion_matrix:
            all_labels = list(LABEL_MAP.values())
            custom_order = sorted(set(all_labels))
            custom_labels = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: custom_order.index(x[1]))]
            cm = confusion_matrix(all_targets, all_preds, labels=custom_order)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title("Confusion Matrix")
            plt.show()

    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    })
    return metrics

def if __name__ == '__main__':
    # Update these paths before running training
    train_faster_rcnn(
        train_img_dir=r"./train_img",
        train_xml_dir=r"./train_xml",
        test_img_dir=r"./val_img",
        test_xml_dir=r"./val_xml",
        num_epochs=50,
        save_path='0508_mobilenetv3_2.pth',
        excel_path='0508_mobilenetv3_2.xlsx'
    )


    backbone = mobilenet_v3_large(pretrained=True).features
    backbone.out_channels = 960

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                      aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes=len(LABEL_MAP) + 1,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    model.to(device)

    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=0.0001)

    train_dataset = CustomDataset(train_img_dir, train_xml_dir)
    test_dataset = CustomDataset(test_img_dir, test_xml_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 初始化 history 字典
    history = {
        'epoch': [], 'mAP': [], 'mAP@50': [], 'mAP@75': [],
        'loss': [], 'cls_loss': [], 'box_loss': [], 'obj_loss': [], 'rpn_box_loss': [],
        'precision': [], 'recall': [], 'f1': [], 'kl_loss': [], 'dist_loss': [],
        'top1_conf': [], 'fdi_error': [], 'epoch_time_sec': [], 'total_time_sec': [],
        'accuracy': []
    }

    def generate_soft_label(gt_label_tensor, device, tau=1.0):
        FDI_INDEX_MAP = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
            9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
            17: 32, 18: 31, 19: 30, 20: 29, 21: 28, 22: 27, 23: 26, 24: 25,
            25: 24, 26: 23, 27: 22, 28: 21, 29: 20, 30: 19, 31: 18, 32: 17
        }
        gt_indices = torch.tensor([FDI_INDEX_MAP.get(lbl.item(), 0) for lbl in gt_label_tensor], device=device)
        soft_labels = []
        for gt in gt_indices:
            dists = torch.tensor([abs(gt - FDI_INDEX_MAP.get(i, 0)) for i in range(1, 33)], device=device).float()
            probs = torch.exp(-dists / tau)
            probs = probs / probs.sum()
            soft = torch.cat([torch.tensor([0.0], device=device), probs], dim=0)
            soft_labels.append(soft)
        return torch.stack(soft_labels, dim=0)

    alpha, beta = 0.5, 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        total_obj_loss = 0
        total_rpn_box_loss = 0
        total_kl_loss = 0
        total_dist_loss = 0
        batch_count = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            batch_images = torch.stack(images, dim=0)
            features = model.backbone(batch_images)
            features_dict = {"0": features}
            image_sizes = [img.shape[-2:] for img in images]
            gt_boxes = [t["boxes"] for t in targets]
            roi_feats = model.roi_heads.box_roi_pool(features_dict, gt_boxes, image_sizes)
            roi_feats = model.roi_heads.box_head(roi_feats)
            logits = model.roi_heads.box_predictor.cls_score(roi_feats)
            true_labels = torch.cat([t["labels"] for t in targets], dim=0)

            if logits.shape[0] > 0 and true_labels.shape[0] > 0:
                log_probs = F.log_softmax(logits, dim=1)
                soft_targets = generate_soft_label(true_labels, device=logits.device)
                kl_loss = KLDivLoss(reduction='batchmean')(log_probs, soft_targets)

                FDI_INDEX_MAP = {
                    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                    9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                    17: 32, 18: 31, 19: 30, 20: 29, 21: 28, 22: 27, 23: 26, 24: 25,
                    25: 24, 26: 23, 27: 22, 28: 21, 29: 20, 30: 19, 31: 18, 32: 17
                }
                pred_class = logits.argmax(dim=1)
                pred_indices = torch.tensor([FDI_INDEX_MAP.get(p.item(), 0) for p in pred_class], device=logits.device)
                true_indices = torch.tensor([FDI_INDEX_MAP.get(t.item(), 0) for t in true_labels], device=logits.device)
                dist_loss = torch.log1p(torch.abs(pred_indices - true_indices).float()).mean()

                kv_loss = alpha * kl_loss + beta * dist_loss
                total_batch_loss = sum(loss_dict.values()) + kv_loss
            else:
                kl_loss = torch.tensor(0.0, device=device)
                dist_loss = torch.tensor(0.0, device=device)
                kv_loss = torch.tensor(0.0, device=device)
                total_batch_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # 累計損失
            total_loss += total_batch_loss.item()
            total_cls_loss += loss_dict.get('loss_classifier', torch.tensor(0.0, device=device)).item()
            total_box_loss += loss_dict.get('loss_box_reg', torch.tensor(0.0, device=device)).item()
            total_obj_loss += loss_dict.get('loss_objectness', torch.tensor(0.0, device=device)).item()
            total_rpn_box_loss += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0, device=device)).item()
            total_kl_loss += kl_loss.item()
            total_dist_loss += dist_loss.item()
            batch_count += 1

        # 計算平均損失
        avg_loss = total_loss / batch_count
        avg_cls_loss = total_cls_loss / batch_count
        avg_box_loss = total_box_loss / batch_count
        avg_obj_loss = total_obj_loss / batch_count
        avg_rpn_box_loss = total_rpn_box_loss / batch_count
        avg_kl_loss = total_kl_loss / batch_count
        avg_dist_loss = total_dist_loss / batch_count

        # 評估模型
        show_cm = (epoch + 1 == num_epochs)
        metrics = evaluate_model(model, test_loader, device, 0.5, show_cm)

        # 記錄指標
        history['epoch'].append(epoch + 1)
        history['mAP'].append(float(metrics['map']))
        history['mAP@50'].append(float(metrics['map_50']))
        history['mAP@75'].append(float(metrics['map_75']))
        history['precision'].append(float(metrics['precision']))
        history['recall'].append(float(metrics['recall']))
        history['f1'].append(float(metrics['f1_score']))
        history['accuracy'].append(float(metrics['accuracy']))
        history['loss'].append(float(avg_loss))
        history['cls_loss'].append(float(avg_cls_loss))
        history['box_loss'].append(float(avg_box_loss))
        history['obj_loss'].append(float(avg_obj_loss))
        history['rpn_box_loss'].append(float(avg_rpn_box_loss))
        history['kl_loss'].append(float(avg_kl_loss))
        history['dist_loss'].append(float(avg_dist_loss))

        # 計算 top1_conf 和 fdi_error
        model.eval()
        total_top1_conf = []
        total_fdi_error = []
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                batch_images = torch.stack(images, dim=0)
                features = model.backbone(batch_images)
                features_dict = {"0": features}
                image_sizes = [img.shape[-2:] for img in images]
                gt_boxes = [t["boxes"] for t in targets]
                roi = model.roi_heads.box_roi_pool(features_dict, gt_boxes, image_sizes)
                feats = model.roi_heads.box_head(roi)
                logits = model.roi_heads.box_predictor.cls_score(feats)
                if logits.shape[0] > 0:
                    preds = logits.argmax(dim=1)
                    labels = torch.cat([t["labels"] for t in targets], dim=0)
                    min_len = min(len(preds), len(labels))
                    preds = preds[:min_len]
                    labels = labels[:min_len]
                    FDI_INDEX_MAP = {
                        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                        9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                        17: 32, 18: 31, 19: 30, 20: 29, 21: 28, 22: 27, 23: 26, 24: 25,
                        25: 24, 26: 23, 27: 22, 28: 21, 29: 20, 30: 19, 31: 18, 32: 17
                    }
                    fdi_pred = [FDI_INDEX_MAP.get(p.item(), 0) for p in preds]
                    fdi_true = [FDI_INDEX_MAP.get(t.item(), 0) for t in labels]
                    if len(fdi_pred) == len(fdi_true) and len(fdi_pred) > 0:
                        fdi_error = sum(abs(p - t) for p, t in zip(fdi_pred, fdi_true)) / len(fdi_pred)
                        total_fdi_error.append(fdi_error)

                outputs = model(images)
                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"]
                    pred_labels = output["labels"]
                    pred_scores = output["scores"]
                    gt_boxes = targets[i]["boxes"]
                    gt_labels = targets[i]["labels"]
                    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                        continue
                    iou_matrix = box_iou(pred_boxes, gt_boxes)
                    matched_pred_indices, matched_gt_indices = [], []
                    for pred_idx, gt_idx in zip(*torch.where(iou_matrix > 0.5)):
                        pred_idx = pred_idx.item()
                        gt_idx = gt_idx.item()
                        if pred_idx not in matched_pred_indices and gt_idx not in matched_gt_indices:
                            matched_pred_indices.append(pred_idx)
                            matched_gt_indices.append(gt_idx)
                    true_positive_scores = []
                    for p_idx, g_idx in zip(matched_pred_indices, matched_gt_indices):
                        if pred_labels[p_idx] == gt_labels[g_idx]:
                            true_positive_scores.append(pred_scores[p_idx].item())
                    if true_positive_scores:
                        top1_conf = sum(true_positive_scores) / len(true_positive_scores)
                        total_top1_conf.append(top1_conf)

        history['fdi_error'].append(sum(total_fdi_error) / len(total_fdi_error) if total_fdi_error else 0.0)
        history['top1_conf'].append(sum(total_top1_conf) / len(total_top1_conf) if total_top1_conf else 0.0)

        # 記錄時間
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time = epoch_end_time - total_start_time
        history['epoch_time_sec'].append(epoch_time)
        history['total_time_sec'].append(total_time)

        # 打印當前 epoch 的結果
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, "
              f"Box Loss: {avg_box_loss:.4f}, Obj Loss: {avg_obj_loss:.4f}, RPN Box Loss: {avg_rpn_box_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}, Dist Loss: {avg_dist_loss:.4f}, "
              f"mAP: {metrics['map']:.4f}, mAP@50: {metrics['map_50']:.4f}, mAP@75: {metrics['map_75']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
              f"FDI Error: {history['fdi_error'][-1]:.4f}, Top1 Conf: {history['top1_conf'][-1]:.4f}, "
              f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")

    # 保存模型和指標
    torch.save(model.state_dict(), save_path)
    pd.DataFrame(history).apply(pd.to_numeric, errors='coerce').to_excel(excel_path, index=False)
    print("Training complete.")

if __name__ == '__main__':
    # Update these paths before running training
    train_faster_rcnn(
        train_img_dir=r"./train_img",
        train_xml_dir=r"./train_xml",
        test_img_dir=r"./val_img",
        test_xml_dir=r"./val_xml",
        num_epochs=50,
        save_path='0508_mobilenetv3_2.pth',
        excel_path='0508_mobilenetv3_2.xlsx'
    )
