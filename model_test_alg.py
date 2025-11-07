import os
import torch
import torchvision
from torchvision.ops import nms, box_iou
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics.detection import MeanAveragePrecision
import torch.nn as nn
from torchvision.models.convnext import convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import numpy as np  # 新增 numpy 來進行數值計算

# --- Configuration (please update paths before running) ---
MODEL_PATH = r"C:\path\to\your\model_best.pth"  # <-- update
VAL_IMG_DIR = r"./val_img"  # <-- update
VAL_XML_DIR = r"./val_xml"  # <-- update
NMS_IOU_THRESHOLD = 0.4
BATCH_SIZE = 2


# ---- LABEL_MAP ----
LABEL_MAP = {
    '18': 1, '17': 2, '16': 3, '15': 4, '14': 5, '13': 6, '12': 7, '11': 8,
    '21': 9, '22': 10, '23': 11, '24': 12, '25': 13, '26': 14, '27': 15, '28': 16,
    '48': 17, '47': 18, '46': 19, '45': 20, '44': 21, '43': 22, '42': 23, '41': 24,
    '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32
}

class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super(ConvNeXtBackbone, self).__init__()
        base_model = convnext_tiny(weights="DEFAULT")
        self.backbone = base_model.features
        self.out_channels = 768

    def forward(self, x):
        return self.backbone(x)

class DentalNet(nn.Module):
    def __init__(self, num_classes):
        super(DentalNet, self).__init__()
        self.backbone = ConvNeXtBackbone()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

        self.detector = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(768, 768 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(768 // 16, 768, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)


# ---- 資料處理、推論與評估流程保持不變 ----
# 請在此段下方接續原本 parse_voc_xml, CustomDataset, 評估等程式碼。
# 並將 model = DentalNet(num_classes=33) 與 model.load_state_dict(...) 保留
# 即可完成與原模型一致的測試版本。

# ---- 資料讀取 ----
def correct_label_box_pair(labels, boxes):
    paired = list(zip(labels, boxes))
    paired.sort(key=lambda x: x[1][0])
    corrected_labels = [p[0] for p in paired]
    corrected_boxes = [p[1] for p in paired]
    return torch.tensor(corrected_labels), torch.tensor(corrected_boxes)

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in LABEL_MAP:
            continue
        bbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = [int(bbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(LABEL_MAP[label])
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].rsplit('.', 1)[0] + '.xml')
        img = Image.open(img_path).convert("RGB")
        boxes, labels = parse_voc_xml(label_path)
        target = {'boxes': boxes, 'labels': labels}
        img = torchvision.transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

# ---- 缺牙檢查函式 ----
def check_missing_teeth(sorted_boxes):
    missing_teeth = []
    for i in range(len(sorted_boxes) - 2):
        x1_max = sorted_boxes[i][2]
        x2_min = sorted_boxes[i + 1][0]
        x2_max = sorted_boxes[i + 1][2]
        if (x2_min - x1_max) > (x2_max - x2_min) / 3:
            missing_teeth.append((i))
    return missing_teeth

# ---- 牙位更正函式（確保label和box一致版本） ----
# ---- 牙位更正函式 ----
def array_with_boxes(labelarray, boxarray):
    upper = lower = maxmatch = index = 0
    array1 = list(range(1, 17))   # 上排牙位 1~16
    array2 = list(range(17, 33))  # 下排牙位 17~32

    # Step 1: 按照 x_min 進行排序
    paired = list(zip(labelarray, boxarray))
    paired.sort(key=lambda x: x[1][0])
    sorted_labels = [p[0] for p in paired]
    sorted_boxes = [p[1] for p in paired]

    # Step 2: 插入缺牙 (label=0, box=[0,0,0,0])
    missing_idx = check_missing_teeth(sorted_boxes)
    insert_shift = 0
    for mi in missing_idx:
        insert_at = mi + 1 + insert_shift
        sorted_labels.insert(insert_at, 0)
        sorted_boxes.insert(insert_at, [0, 0, 0, 0])
        insert_shift += 1

    # Step 3: 判斷牙列是上排還是下排
    for lbl in sorted_labels:
        if lbl <= 16 and lbl > 0:
            upper += 1
        elif lbl > 0:
            lower += 1

    n = len(sorted_labels)
    best_match_array = array1 if upper > lower else array2

    # Step 4: 找滑動對齊起點（最大 match）
    for i in range(len(best_match_array) - n + 1):
        match = 0
        for j in range(n):
            if sorted_labels[j] == best_match_array[i + j]:
                match += 1
        if match > maxmatch:
            maxmatch = match
            index = i

    # Step 5: 若 match > 1 才進行牙位更正，否則維持原始 label
    if maxmatch > 1:
        corrected_labels = []
        for i in range(n):
            if sorted_labels[i] == 0:
                corrected_labels.append(0)
            else:
                corrected_labels.append(best_match_array[index + i])
    else:
        corrected_labels = sorted_labels

    # Step 6: 將缺牙 (label=0) 移除
    final_labels = []
    final_boxes = []
    for lbl, box in zip(corrected_labels, sorted_boxes):
        if lbl != 0:
            final_labels.append(lbl)
            final_boxes.append(box)

    return final_labels, final_boxes

# ---- 解析 XML ----


def collate_fn(batch):
    return tuple(zip(*batch))

# ---- 測試與評估 ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DentalNet(num_classes=33)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

val_dataset = CustomDataset(
    VAL_IMG_DIR,
    VAL_XML_DIR
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---- 評估 ----
mAP_metric = MeanAveragePrecision(iou_thresholds=[0.5 + i * 0.05 for i in range(10)])
all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in tqdm(val_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            keep = nms(output['boxes'], output['scores'], 0.4)
            pred_boxes = output['boxes'][keep].cpu().tolist()
            pred_labels = output['labels'][keep].cpu().tolist()
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']

            # 牙位更正並保持label與box一一對應
            corrected_labels, corrected_boxes = array_with_boxes(pred_labels, pred_boxes)
            corrected_labels = torch.tensor(corrected_labels)
            corrected_boxes = torch.tensor(corrected_boxes)

            if len(corrected_labels) > 0 and len(target_labels) > 0:
                pred_dict = {'boxes': corrected_boxes, 'scores': output['scores'][keep].cpu(), 'labels': corrected_labels}
                target_dict = {'boxes': target_boxes, 'labels': target_labels}
                mAP_metric.update([pred_dict], [target_dict])

                iou_matrix = box_iou(target_boxes, corrected_boxes)
                for t_idx, p_idx in zip(*torch.where(iou_matrix > 0.5)):
                    all_targets.append(target_labels[t_idx].item())
                    all_preds.append(corrected_labels[p_idx].item())

# ---- 結果 ----
metrics = mAP_metric.compute()
precision, recall, f1 = precision_recall_fscore_support(all_targets, all_preds, average='weighted')[:3]

print("mAP (All):", metrics['map'].item())
print("mAP@50:", metrics['map_50'].item())
print("mAP@75:", metrics['map_75'].item())
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

if len(all_targets) > 0:
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(1, 33)))
    
    # 計算每個類別的 Specificity
    specificities = []
    class_counts = np.sum(cm, axis=1)  # 每個類別的真實樣本數
    total_samples = np.sum(class_counts)  # 總樣本數
    
    for i in range(len(cm)):
        # 真陰性（TN）：除了該類別的行和列之外的所有元素之和
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # 假陽性（FP）：該類別的列總和（預測為該類別）減去對角線（正確預測）
        fp = np.sum(cm[:, i]) - cm[i, i]
        # Specificity = TN / (TN + FP)
        if (tn + fp) > 0:  # 避免除以 0
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
        specificities.append(specificity)
    
    # 計算加權平均 Specificity（根據每個類別的真實樣本數加權）
    weighted_specificity = np.sum(np.array(specificities) * class_counts) / total_samples if total_samples > 0 else 0.0
    
    print("Weighted Specificity:", weighted_specificity)

# ---- 混淆矩陣 ----
if len(all_targets) > 0:
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(1, 33)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(k) for k in range(1, 33)])
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix (Corrected + IoU matched)")
    plt.show()

if __name__ == '__main__':
    # basic checks
    if not Path(MODEL_PATH).exists():
        print(f"[Warning] MODEL_PATH does not exist: {MODEL_PATH}\nPlease update MODEL_PATH in the configuration section.")
    if not Path(VAL_IMG_DIR).exists():
        print(f"[Warning] VAL_IMG_DIR does not exist: {VAL_IMG_DIR}\nPlease update VAL_IMG_DIR in the configuration section.")
    if not Path(VAL_XML_DIR).exists():
        print(f"[Warning] VAL_XML_DIR does not exist: {VAL_XML_DIR}\nPlease update VAL_XML_DIR in the configuration section.")
