# -*- coding: utf-8 -*-
"""
Tạo file .pt từ TOÀN BỘ Dataset_1 (không split train/val)
dùng để test cross-dataset ở bảng 9

Dataset_1 cấu trúc:
  rp_im/tr_im.nii               — CT scans (4D: H x W x slices)
  tr_lung_mask/tr_lungmasks_updated.nii  — lung masks
  tr_mask.nii/tr_mask.nii       — infection masks
"""

import os
import numpy as np
import cv2
import nibabel as nib
import torch

database_pathh = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\data\9CTscans\Dataset1'

slice_path = os.path.join(database_pathh, 'rp_im',        'tr_im.nii')
mask_path  = os.path.join(database_pathh, 'tr_lung_mask', 'tr_lungmasks_updated.nii')
inf_path   = os.path.join(database_pathh, 'tr_mask.nii',  'tr_mask.nii')

slices = nib.load(slice_path).get_fdata()
masks  = nib.load(mask_path).get_fdata()
infs   = nib.load(inf_path).get_fdata()

print(f'slices shape: {slices.shape}')
print(f'masks  shape: {masks.shape}')
print(f'infs   shape: {infs.shape}')

All_data = []
All_lung = []
All_inf  = []

# Axis 2 = slice dimension
for i in range(infs.shape[2]):
    slice1 = cv2.rotate(slices[:, :, i], cv2.ROTATE_90_CLOCKWISE)
    mask1  = np.uint8(cv2.rotate(masks[:, :, i], cv2.ROTATE_90_CLOCKWISE))
    inf1   = np.uint8(cv2.rotate(infs[:, :, i],  cv2.ROTATE_90_CLOCKWISE))

    img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    All_data.append(img)
    All_lung.append(mask1)
    All_inf.append(inf1)

print(f'Total slices in Dataset_1: {len(All_data)}')

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset1_full.pt')
torch.save((All_data, All_lung, All_inf), save_path)
print(f'Saved to: {save_path}')
