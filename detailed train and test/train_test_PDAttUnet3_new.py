# -*- coding: utf-8 -*-
"""
train_test_PDAttUnet3_new.py
Script train mô hình PDAtt-Unet trên Dataset 3.
Lưu vào thư mục: Model_dataset3_new/PYAttUNet/Models/...
Hàm Loss sử dụng gamma = 2.0 theo quy định của bài báo.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks
from torch.utils.data import Dataset

class Data_loaderVE(Dataset):
    def __init__(self, path, transform=None):
        self.data, self.y, self.y2 = torch.load(path)
        self.transform = transform

    def __getitem__(self, index):
        img1 = np.array(self.data[index]).astype(np.uint8)
        y1 = np.array(self.y[index]).astype(np.uint8)
        y2 = np.array(self.y2[index]).astype(np.uint8)

        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0
        
        y33 = (y2 * 255.0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
        y3[y3 > 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2, y3])
            img1 = augmentations["image"]
            y1 = augmentations["masks"][0]
            y2 = augmentations["masks"][1]
            y3 = augmentations["masks"][2]

        return img1, y2, y1, y3

    def __len__(self):
        return len(self.data)

############################
img_size = 224

train_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

############################
base_dir = os.path.dirname(os.path.abspath(__file__))
train_pt = os.path.join(base_dir, 'Training_data3.pt')
val_pt   = os.path.join(base_dir, 'Validation_data3.pt')

if not os.path.exists(train_pt) or not os.path.exists(val_pt):
    print(f"Error: {train_pt} or {val_pt} does not exist.")
    print("Please run `python prepare_dataset3_split.py` first!")
    sys.exit()

train_set = Data_loaderVE(path=train_pt, transform=train_transform)
validate_set = Data_loaderVE(path=val_pt, transform=val_transforms)

F1_mean, dise_mean, IoU_mean = [], [], []

modl_n = 'PYAttUNet'
epochs = 60
batch_size = 6
runs = 5

for itr in range(runs):
    model_sp = os.path.join(base_dir, "Model_dataset3_new", modl_n, "Models")
    os.makedirs(model_sp, exist_ok=True)

    name_model_final = os.path.join(model_sp, f"{itr}_fi.pt")
    name_model_bestF1 = os.path.join(model_sp, f"{itr}_bt.pt")

    model_spR = os.path.join(base_dir, "Model_dataset3_new", modl_n, "Results")
    os.makedirs(model_spR, exist_ok=True)

    training_tsx = os.path.join(model_spR, f"{itr}.txt")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = getattr(networks, modl_n)()
    model.to(device)
    
    segm = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()

    train_acc, train_F1score, train_dise, train_IoU, train_Sens, train_Spec, train_Prec = [], [], [], [], [], [], []
    valid_acc, valid_F1score, valid_dise, valid_IoU, valid_Sens, valid_Spec, valid_Prec = [], [], [], [], [], [], []

    best_F1score = 0.0
    best_index = 0

    print(f"\n====================== RUN {itr} ======================")
    for epoch in range(epochs):
        lr = 0.01
        if epoch > 30: lr = 0.001
        if epoch > 50: lr = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        dice_scores, TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0, 0.0
        num_correct, num_pixels = 0, 0
        
        for batch in tqdm(train_loader):
            x, y, y_lung, y_edge = batch
            x = x.to(device)
            y = y.float().to(device)
            y_lung = y_lung.float().to(device)
            y_edge = y_edge.float().to(device)

            optimizer.zero_grad()
            outputs, outputs2 = model(x)
            
            # Hybrid Loss
            loss1 = criterion(outputs.squeeze(dim=1), y)
            if outputs2 is not None:
                loss2 = criterion(outputs2.squeeze(dim=1), y_lung)
                # Edge loss uses the predicted infection mask (outputs) weighted by the ground truth edge
                # However, since outputs are logits, we apply sigmoid before multiplying with edge
                # Wait, the original code used outputs directly. Let's strictly use the exact same formula.
                loss3 = criterion(outputs.squeeze(dim=1) * y_edge, y_edge)
                loss = 0.7 * loss1 + 0.3 * loss2 + 2.0 * loss3  # GAMMA = 2.0 for Dataset 3
            else:
                loss = loss1

            loss.backward()
            optimizer.step()

            preds = (segm(outputs) > 0.5).squeeze(1).cpu().numpy().astype(int)
            yy = y.squeeze(1).cpu().numpy().astype(int)

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

        epoch_dise = dice_scores / len(train_set)
        epoch_acc = (num_correct / num_pixels) * 100
        Spec = 1 - (FP / (FP + TN + 1e-8))
        Sens = TP / (TP + FN + 1e-8)
        Prec = TP / (TP + FP + 1e-8)
        F1score = TP / (TP + 0.5 * (FP + FN) + 1e-8)
        IoU = TP / (TP + FP + FN + 1e-8)

        train_acc.append(epoch_acc)
        train_F1score.append(F1score)
        train_dise.append(epoch_dise)
        train_IoU.append(IoU)
        train_Sens.append(Sens)
        train_Spec.append(Spec)
        train_Prec.append(Prec)

        print(f"Epoch {epoch}/{epochs-1}")
        print("-" * 10)
        print(f"train Acc: {epoch_acc:.4f} Dise: {epoch_dise:.4f} IoU: {IoU:.4f} F1: {F1score:.4f} Sens: {Sens:.4f} Prec: {Prec:.4f}")
        
        with open(training_tsx, "a") as f:
            print(f"Epoch {epoch}/{epochs-1}", file=f)
            print("-" * 10, file=f)
            print(f"train Acc: {epoch_acc:.8f} Dise: {epoch_dise:.8f} IoU: {IoU:.8f} F1: {F1score:.8f} Spec: {Spec:.8f} Sens: {Sens:.8f} Prec: {Prec:.8f}", file=f)

        # VALIDATION
        model.eval()
        dice_scores, TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0, 0.0
        num_correct, num_pixels = 0, 0

        with torch.no_grad():
            for batch in tqdm(validate_loader):
                x, y, y_lung, y_edge = batch
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

        valid_acc.append(epoch_acc)
        valid_F1score.append(F1score)
        valid_dise.append(epoch_dise)
        valid_IoU.append(IoU)
        valid_Sens.append(Sens)
        valid_Spec.append(Spec)
        valid_Prec.append(Prec)

        if F1score > best_F1score:
            best_F1score = F1score
            best_index = epoch
            torch.save(model.state_dict(), name_model_bestF1)

        print(f"valid Acc: {epoch_acc:.4f} Dise: {epoch_dise:.4f} IoU: {IoU:.4f} F1: {F1score:.4f} Sens: {Sens:.4f} Prec: {Prec:.4f}")
        
        with open(training_tsx, "a") as f:
            print(f"Epoch {epoch}/{epochs-1}", file=f)
            print("-" * 10, file=f)
            print(f"valid Acc: {epoch_acc:.8f} Dise: {epoch_dise:.8f} IoU: {IoU:.8f} F1: {F1score:.8f} Spec: {Spec:.8f} Sens: {Sens:.8f} Prec: {Prec:.8f}", file=f)

    torch.save(model.state_dict(), name_model_final)

    with open(training_tsx, "a") as f:
        print("Train F1 score", file=f)
        print(train_F1score[-1], file=f)
        print("Best Val F1 score", file=f)
        print(best_F1score, file=f)
        print("Index of Best", file=f)
        print(name_model_bestF1, file=f)
        print(best_index, file=f)

    print(f"Best Val F1: {best_F1score:.4f} at epoch {best_index}")

    F1_mean.append(best_F1score)
    dise_mean.append(valid_dise[best_index])
    IoU_mean.append(valid_IoU[best_index])

std1 = np.std(F1_mean)
std2 = np.std(dise_mean)
std3 = np.std(IoU_mean)

mean_path = os.path.join(base_dir, "Model_dataset3_new", modl_n, "Results", "mean.txt")
with open(mean_path, "w") as f:
    print(f"F1_mean  {F1_mean + [np.mean(F1_mean), std1]}", file=f)
    print(f"dise_mean {dise_mean + [np.mean(dise_mean), std2]}", file=f)
    print(f"IoU_mean  {IoU_mean + [np.mean(IoU_mean), std3]}", file=f)

print("\n========== FINAL RESULTS (Dataset 3 - NEW) ==========")
print(f"F1  : {[round(float(v),4) for v in F1_mean]}  mean={np.mean(F1_mean):.4f} ±{std1:.4f}")
print(f"Dice: {[round(float(v),4) for v in dise_mean]}  mean={np.mean(dise_mean):.4f} ±{std2:.4f}")
print(f"IoU : {[round(float(v),4) for v in IoU_mean]}  mean={np.mean(IoU_mean):.4f} ±{std3:.4f}")
