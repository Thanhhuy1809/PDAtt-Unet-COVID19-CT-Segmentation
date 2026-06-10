# -*- coding: utf-8 -*-
"""
Table 9: Test models trained on Dataset 2 on the FULL Dataset 1
- Models: Model_dataset2/PYAttUNet/Models/{i}_bt.pt  (5 runs, take mean)
- Test data: Training_data1.pt + Validation_data1.pt (ALL of Dataset 1)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks

class SingleDataset(Dataset):
    """Load 1 file .pt để test"""
    def __init__(self, path, transform=None):
        self.data, self.y, self.y2 = torch.load(path)
        self.transform = transform

    def __getitem__(self, index):
        img1, y1, y2 = self.data[index], self.y[index], self.y2[index]  # y1: Lung, y2: Infection

        img1 = np.array(img1)
        y1 = np.array(y1)
        y2 = np.array(y2)
        img1 = img1.astype(np.uint8)
        y1 = y1.astype(np.uint8)
        y2 = y2.astype(np.uint8)

        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0
        y33 = y2 * 255.0
        y33 = y33.astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
        y3[y3 > 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2, y3])
            image = augmentations["image"]
            mask = augmentations["masks"][0]
            mask1 = augmentations["masks"][1]
            edge = augmentations["masks"][2]

        return image, mask1, mask, edge

    def __len__(self):
        return len(self.data)


##########################################
# Config
##########################################
img_size = 224

val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

base_dir = os.path.dirname(os.path.abspath(__file__))

# FULL Dataset_1 từ prepare_Dataset2_full.py
Dataset2_path = os.path.join(base_dir, 'Dataset2_full.pt')

# Models trained on Dataset 2
model_src = os.path.join(base_dir, 'Model_dataset2', 'PYAttUNet', 'Models')

# Results output
result_dir = os.path.join(base_dir, 'Model_D2_test_D2', 'PYAttUNet', 'Results')
os.makedirs(result_dir, exist_ok=True)

modl_n = 'PYAttUNet'
runs = 5  # chạy cả 5 model, lấy mean như trong paper

##########################################
# Load FULL Dataset_1
##########################################
validate_set = SingleDataset(path=Dataset2_path, transform=val_transforms)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=30, shuffle=False)
print(f'Total test samples (full Dataset_1): {len(validate_set)}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
segm = nn.Sigmoid()

F1_mean, dise_mean, IoU_mean = [], [], []

##########################################
# Test loop: 5 models từ Dataset 2 → mean như trong paper
##########################################
for itr in range(runs):
    model_path = os.path.join(model_src, f'{itr}_bt.pt')
    result_path = os.path.join(result_dir, f'{itr}.txt')

    print(f'\n--- Run {itr} | Loading: {model_path} ---')

    model = getattr(networks, modl_n)()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dice_scores = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    num_correct = 0
    num_pixels = 0

    with torch.no_grad():
        for batch in tqdm(validate_loader):
            x, y, y2, edge = batch
            x = x.to(device)
            y = y.float().to(device)

            outputs, outputs2 = model(x)

            preds = segm(outputs) > 0.5
            preds = preds.squeeze(dim=1).cpu().numpy().astype(int)
            yy = y > 0.5
            yy = yy.squeeze(dim=1).cpu().numpy().astype(int)

            num_correct += np.sum(preds == yy)
            num_pixels += preds.size

            TP += np.sum(((preds == 1).astype(int) + (yy == 1).astype(int)) == 2)
            TN += np.sum(((preds == 0).astype(int) + (yy == 0).astype(int)) == 2)
            FP += np.sum(((preds == 1).astype(int) + (yy == 0).astype(int)) == 2)
            FN += np.sum(((preds == 0).astype(int) + (yy == 1).astype(int)) == 2)

            for idice in range(preds.shape[0]):
                dice_scores += (2 * (preds[idice] * yy[idice]).sum()) / (
                    (preds[idice] + yy[idice]).sum() + 1e-8
                )


    epoch_dise = dice_scores / len(validate_loader.dataset)
    epoch_acc2 = (num_correct / num_pixels) * 100

    Spec    = 1 - (FP / (FP + TN))
    Sens    = TP / (TP + FN)
    Prec    = TP / (TP + FP + 1e-8)
    F1score = TP / (TP + ((1 / 2) * (FP + FN)) + 1e-8)
    IoU     = TP / (TP + FP + FN)

    with open(result_path, "w") as f:
        print(f'Run {itr} | D2 -> D2 (full)', file=f)
        print('-' * 10, file=f)
        print(f'Acc:  {epoch_acc2:.8f}', file=f)
        print(f'Dise: {epoch_dise:.8f}', file=f)
        print(f'IoU:  {IoU:.8f}', file=f)
        print(f'F1:   {F1score:.8f}', file=f)
        print(f'Sens: {Sens:.8f}', file=f)
        print(f'Spec: {Spec:.8f}', file=f)
        print(f'Prec: {Prec:.8f}', file=f)

    print(f'  F1={float(F1score):.4f}  Dice={float(epoch_dise):.4f}  IoU={float(IoU):.4f}')

    F1_mean.append(F1score)
    dise_mean.append(epoch_dise)
    IoU_mean.append(IoU)

##########################################
# Mean & Std (như paper)
##########################################
import numpy as np
stD2 = np.std(F1_mean)
std2 = np.std(dise_mean)
std3 = np.std(IoU_mean)

mean_path = os.path.join(result_dir, 'mean.txt')
with open(mean_path, "w") as f:
    print(f'F1_mean  {F1_mean + [np.mean(F1_mean), stD2]}', file=f)
    print(f'dise_mean {dise_mean + [np.mean(dise_mean), std2]}', file=f)
    print(f'IoU_mean  {IoU_mean + [np.mean(IoU_mean), std3]}', file=f)

print('\n========== FINAL RESULTS (Table 9 - D2 model on full D2) ==========')
print(f'F1  : {[round(float(v),4) for v in F1_mean]}  mean={np.mean(F1_mean):.4f} ±{stD2:.4f}')
print(f'Dice: {[round(float(v),4) for v in dise_mean]}  mean={np.mean(dise_mean):.4f} ±{std2:.4f}')
print(f'IoU : {[round(float(v),4) for v in IoU_mean]}  mean={np.mean(IoU_mean):.4f} ±{std3:.4f}')
print(f'Results saved to: {result_dir}')
