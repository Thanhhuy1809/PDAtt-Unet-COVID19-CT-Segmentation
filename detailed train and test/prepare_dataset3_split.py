# -*- coding: utf-8 -*-
"""
prepare_dataset3_split.py
Tạo Training_data3.pt và Validation_data3.pt từ Dataset 3 (20 bệnh nhân).
Sử dụng Patient-independent splitting (70% bệnh nhân cho Train, 30% cho Val).
"""

import os
import numpy as np
import cv2
import nibabel as nib
import torch
from sklearn.model_selection import train_test_split

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\data\9CTscans\Dataset3'
CT_DIR   = os.path.join(BASE_DIR, 'COVID-19-CT-Seg_20cases')
INF_DIR  = os.path.join(BASE_DIR, 'Infection_Mask')
LUNG_DIR = os.path.join(BASE_DIR, 'Lung_Mask')

OUT_DIR = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test'
TRAIN_PT = os.path.join(OUT_DIR, 'Training_data3.pt')
VAL_PT   = os.path.join(OUT_DIR, 'Validation_data3.pt')

CASE_MAPPING = [
    ('case009.nii', 'coronacases_009.nii'),
    ('coronacases_org_001.nii', 'coronacases_001.nii'),
    ('coronacases_org_002.nii', 'coronacases_002.nii'),
    ('coronacases_org_003.nii', 'coronacases_003.nii'),
    ('coronacases_org_004.nii', 'coronacases_004.nii'),
    ('coronacases_org_005.nii', 'coronacases_005.nii'),
    ('coronacases_org_006.nii', 'coronacases_006.nii'),
    ('coronacases_org_008.nii', 'coronacases_008.nii'),
    ('coronacases_org_case007.nii', 'coronacases_007.nii'),
    ('coronacases_org_case010.nii', 'coronacases_010.nii'),
    ('radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii', 'radiopaedia_10_85902_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii', 'radiopaedia_10_85902_3.nii'),
    ('radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii', 'radiopaedia_14_85914_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii', 'radiopaedia_27_86410_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm.nii', 'radiopaedia_29_86490_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii', 'radiopaedia_29_86491_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm.nii', 'radiopaedia_36_86526_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.nii', 'radiopaedia_40_86625_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii', 'radiopaedia_4_85506_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii', 'radiopaedia_7_85703_0.nii'),
]

# Chia 70% Train, 30% Val theo Bệnh nhân (Patient-Independent)
indicies = list(range(len(CASE_MAPPING)))
train_spt, val_spt = train_test_split(indicies, test_size=0.3, random_state=42)

def process_and_save(split_indices, output_path, split_name):
    print(f"\nProcessing {split_name} split...")
    data, lung, inf = [], [], []
    for idx in split_indices:
        ct_fname, mask_fname = CASE_MAPPING[idx]
        ct_path   = os.path.join(CT_DIR, ct_fname)
        inf_path  = os.path.join(INF_DIR, mask_fname)
        lung_path = os.path.join(LUNG_DIR, mask_fname)
        
        if not all([os.path.exists(ct_path), os.path.exists(inf_path), os.path.exists(lung_path)]):
            continue
            
        slices = nib.load(ct_path).get_fdata()
        infs   = nib.load(inf_path).get_fdata()
        masks  = nib.load(lung_path).get_fdata()
        
        n_slices = slices.shape[2]
        print(f"  {ct_fname} ({n_slices} slices)")
        
        for i in range(n_slices):
            s = cv2.rotate(slices[:, :, i], cv2.ROTATE_90_CLOCKWISE)
            m = np.uint8(np.round(cv2.rotate(masks[:, :, i], cv2.ROTATE_90_CLOCKWISE)))
            i_mask = np.uint8(np.round(cv2.rotate(infs[:, :, i], cv2.ROTATE_90_CLOCKWISE)))
            
            img = cv2.normalize(s, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            data.append(img)
            lung.append(m)
            inf.append(i_mask)
            
    print(f"Saving {split_name}: {len(data)} total slices...")
    torch.save((data, lung, inf), output_path, _use_new_zipfile_serialization=False)
    print(f"Done saving {output_path}")

process_and_save(train_spt, TRAIN_PT, "TRAIN")
process_and_save(val_spt, VAL_PT, "VALIDATION")
print("\nHoàn tất chuẩn bị Dataset 3!")
