# -*- coding: utf-8 -*-
"""
Table 9: Test models trained on Dataset 3 on the FULL Dataset 2 (cross-dataset)
- Models: Model_dataset3/PYAttUNet/Models/{i}_bt.pt  (5 runs, take mean)
- Test data: Training_data2a.pt + Validation_data2a.pt (FULL Dataset 2)
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

import albumentations as A

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks


class CombinedDataset(Dataset):
    def __init__(self, paths, transform=None):
        data_list = []
        y_list = []
        y2_list = []
        for path in paths:
            data, y, y2 = torch.load(path, weights_only=False)
            data_list.extend(list(data))
            y_list.extend(list(y))
            y2_list.extend(list(y2))

        self.data = data_list
        self.y = y_list
        self.y2 = y2_list
        self.transform = transform

    def __getitem__(self, index):
        img1 = np.array(self.data[index]).astype(np.uint8)
        y1 = np.array(self.y[index]).astype(np.uint8)
        y2 = np.array(self.y2[index]).astype(np.uint8)

        y1[y1 > 0] = 1
        y2[y2 > 0] = 1

        y33 = (y2 * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
        y3[y3 > 0] = 1

        if self.transform is not None:
            aug = self.transform(image=img1, masks=[y1, y2, y3])
            img1 = aug["image"]
            y1 = aug["masks"][0]
            y2 = aug["masks"][1]
            y3 = aug["masks"][2]

        return img1, y2, y1, y3  # image, infection_mask, lung_mask, edge

    def __len__(self):
        return len(self.data)


##########################################
# Config
##########################################
img_size = 224

val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset2_paths = [
    os.path.join(base_dir, "Training_data2a.pt"),
    os.path.join(base_dir, "Validation_data2a.pt"),
]

model_src = os.path.join(base_dir, "Model_dataset3_new", "PYAttUNet", "Models")
result_dir = os.path.join(base_dir, "Model_D3_new_test_D2", "PYAttUNet", "Results")
os.makedirs(result_dir, exist_ok=True)

runs = 5

##########################################
# Load FULL Dataset 2
##########################################
validate_set = CombinedDataset(paths=dataset2_paths, transform=val_transforms)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=30, shuffle=False)
print(f"Total test samples (full Dataset 2): {len(validate_set)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
segm = nn.Sigmoid()

F1_mean, dise_mean, IoU_mean = [], [], []

##########################################
# Test loop: 5 models from Dataset 3
##########################################
for itr in range(runs):
    model_path = os.path.join(model_src, f"{itr}_bt.pt")
    result_path = os.path.join(result_dir, f"{itr}.txt")

    print(f"\n--- Run {itr} | Loading: {model_path} ---")

    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    dice_scores = 0.0
    TP = TN = FP = FN = 0.0
    num_correct = num_pixels = 0

    with torch.no_grad():
        for batch in tqdm(validate_loader):
            x, y, y2, edge = batch
            x = x.to(device)
            y = y.float().to(device)

            outputs, outputs2 = model(x)

            preds = (segm(outputs) > 0.5).squeeze(1).cpu().numpy().astype(int)
            yy = (y > 0.5).squeeze(1).cpu().numpy().astype(int)

            num_correct += np.sum(preds == yy)
            num_pixels += preds.size

            TP += float(((preds == 1) & (yy == 1)).sum())
            TN += float(((preds == 0) & (yy == 0)).sum())
            FP += float(((preds == 1) & (yy == 0)).sum())
            FN += float(((preds == 0) & (yy == 1)).sum())

            for k in range(preds.shape[0]):
                union = (preds[k] + yy[k]).sum()
                if union == 0:
                    dice_scores += 0.0
                else:
                    dice_scores += (2 * (preds[k] * yy[k]).sum()) / (union + 1e-8)

    epoch_dise = dice_scores / len(validate_set)
    epoch_acc = (num_correct / num_pixels) * 100
    Spec = 1 - (FP / (FP + TN + 1e-8))
    Sens = TP / (TP + FN + 1e-8)
    Prec = TP / (TP + FP + 1e-8)
    F1score = TP / (TP + 0.5 * (FP + FN) + 1e-8)
    IoU = TP / (TP + FP + FN + 1e-8)

    with open(result_path, "w") as f:
        print("Run %d | D3 model -> D2 full (cross-dataset)" % itr, file=f)
        print("-" * 10, file=f)
        print("Acc:  %.8f" % epoch_acc, file=f)
        print("Dise: %.8f" % epoch_dise, file=f)
        print("IoU:  %.8f" % IoU, file=f)
        print("F1:   %.8f" % F1score, file=f)
        print("Sens: %.8f" % Sens, file=f)
        print("Spec: %.8f" % Spec, file=f)
        print("Prec: %.8f" % Prec, file=f)

    print("  F1=%.4f  Dice=%.4f  IoU=%.4f" % (float(F1score), float(epoch_dise), float(IoU)))

    F1_mean.append(float(F1score))
    dise_mean.append(float(epoch_dise))
    IoU_mean.append(float(IoU))

##########################################
# Mean & Std
##########################################
std1 = np.std(F1_mean)
std2 = np.std(dise_mean)
std3 = np.std(IoU_mean)

mean_path = os.path.join(result_dir, "mean.txt")
with open(mean_path, "w") as f:
    print("F1_mean   %s" % (F1_mean + [np.mean(F1_mean), std1]), file=f)
    print("dise_mean %s" % (dise_mean + [np.mean(dise_mean), std2]), file=f)
    print("IoU_mean  %s" % (IoU_mean + [np.mean(IoU_mean), std3]), file=f)

print("\n========== FINAL RESULTS (D3 model on full D2, cross-dataset) ==========")
print("F1  : %s   mean=%.4f +-%.4f" % ([round(v, 4) for v in F1_mean], np.mean(F1_mean), std1))
print("Dice: %s   mean=%.4f +-%.4f" % ([round(v, 4) for v in dise_mean], np.mean(dise_mean), std2))
print("IoU : %s   mean=%.4f +-%.4f" % ([round(v, 4) for v in IoU_mean], np.mean(IoU_mean), std3))
print("Results saved to: %s" % result_dir)
