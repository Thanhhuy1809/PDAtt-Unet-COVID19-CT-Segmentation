import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

##########################################   
##########################################  
##########################################
from torch.utils.data import Dataset
import os
import os.path
import numpy as np
import torch
import cv2


class Data_loaderVE(Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train  # training set or test set
        self.data, self.y, self.y2 = torch.load(os.path.join(root, train))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
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

        ############################
img_size = 224

train_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
######
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

############################
# Part 2

train_set = Data_loaderVE(
        root='./Datas'
        ,train = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Training_data2.pt'
        ,transform = train_transform )

validate_set = Data_loaderVE(
        root='./Datas'
        ,train = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Validation_data2.pt'
        ,transform = val_transforms
)

############################
F1_mean, dise_mean, IoU_mean = [], [], []

dataset_idx = 3

modl_n = 'PYAttUNet'