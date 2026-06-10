# -*- coding: utf-8 -*-
"""
prepare_dataset3_full.py

Load all 20 COVID-19 CT cases from Dataset3 and save as Dataset3_full.pt
for cross-dataset evaluation (Table 10 in paper).

Dataset3 structure (after running extract_dataset3.py):
  COVID-19-CT-Seg_20cases/  -> CT images (.nii, tên dạng coronacases_org_*, radiopaedia_org_*)
  Infection_Mask/           -> infection masks (.nii, tên dạng coronacases_*, radiopaedia_*)
  Lung_Mask/                -> lung masks (.nii, tên dạng coronacases_*, radiopaedia_*)

Tên file CT và mask KHÁC NHAU nên dùng mapping rõ ràng bên dưới.
"""

import os
import numpy as np
import cv2
import nibabel as nib
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\data\9CTscans\Dataset3'

CT_DIR   = os.path.join(BASE_DIR, 'COVID-19-CT-Seg_20cases')
INF_DIR  = os.path.join(BASE_DIR, 'Infection_Mask')
LUNG_DIR = os.path.join(BASE_DIR, 'Lung_Mask')

OUTPUT_PT = os.path.join(
    r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test',
    'Dataset3_full.pt'
)

# ── mapping: (ct_filename, mask_filename) ─────────────────────────────────────
# CT files có prefix _org_ hoặc tên đặc biệt; mask files dùng tên chuẩn
CASE_MAPPING = [
    ('case009.nii',                                              'coronacases_009.nii'),
    ('coronacases_org_001.nii',                                  'coronacases_001.nii'),
    ('coronacases_org_002.nii',                                  'coronacases_002.nii'),
    ('coronacases_org_003.nii',                                  'coronacases_003.nii'),
    ('coronacases_org_004.nii',                                  'coronacases_004.nii'),
    ('coronacases_org_005.nii',                                  'coronacases_005.nii'),
    ('coronacases_org_006.nii',                                  'coronacases_006.nii'),
    ('coronacases_org_008.nii',                                  'coronacases_008.nii'),
    ('coronacases_org_case007.nii',                              'coronacases_007.nii'),
    ('coronacases_org_case010.nii',                              'coronacases_010.nii'),
    ('radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii',   'radiopaedia_10_85902_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii',   'radiopaedia_10_85902_3.nii'),
    ('radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii',   'radiopaedia_14_85914_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii',   'radiopaedia_27_86410_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm.nii',   'radiopaedia_29_86490_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii',   'radiopaedia_29_86491_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm.nii',   'radiopaedia_36_86526_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.nii',   'radiopaedia_40_86625_0.nii'),
    ('radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii',    'radiopaedia_4_85506_1.nii'),
    ('radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii',    'radiopaedia_7_85703_0.nii'),
]

# Xóa file cũ bị hỏng (nếu có) trước khi tạo mới
if os.path.exists(OUTPUT_PT):
    os.remove(OUTPUT_PT)
    print(f"Deleted old (corrupt) file: {OUTPUT_PT}")

print(f"Processing {len(CASE_MAPPING)} CT scans in Dataset3")

All_data = []   # images  (H x W x 3, uint8)
All_lung = []   # lung masks (H x W, uint8)
All_inf  = []   # infection masks (H x W, uint8)

for ct_fname, mask_fname in CASE_MAPPING:
    ct_path   = os.path.join(CT_DIR,   ct_fname)
    inf_path  = os.path.join(INF_DIR,  mask_fname)
    lung_path = os.path.join(LUNG_DIR, mask_fname)

    if not os.path.exists(ct_path):
        print(f"  [WARN] CT file not found: {ct_fname}, skipping")
        continue
    if not os.path.exists(inf_path):
        print(f"  [WARN] Infection mask not found: {mask_fname}, skipping")
        continue
    if not os.path.exists(lung_path):
        print(f"  [WARN] Lung mask not found: {mask_fname}, skipping")
        continue

    slices = nib.load(ct_path).get_fdata()
    infs   = nib.load(inf_path).get_fdata()
    masks  = nib.load(lung_path).get_fdata()

    n_slices = slices.shape[2]
    pos_slices = int(np.sum(np.any(infs > 0, axis=(0, 1))))
    print(f"  {ct_fname}: {n_slices} slices, {pos_slices} with infection")

    for i in range(n_slices):
        s = slices[:, :, i]
        s = cv2.rotate(s, cv2.ROTATE_90_CLOCKWISE)

        # dùng np.round trước khi uint8 để tránh float truncation (1.9999 -> 1 thay vì 2)
        m = np.uint8(np.round(cv2.rotate(masks[:, :, i], cv2.ROTATE_90_CLOCKWISE)))
        inf = np.uint8(np.round(cv2.rotate(infs[:, :, i],  cv2.ROTATE_90_CLOCKWISE)))

        img = cv2.normalize(s, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        All_data.append(img)
        All_lung.append(m)
        All_inf.append(inf)

total_inf_slices = sum(1 for x in All_inf if x.max() > 0)
print(f"\nTotal slices       : {len(All_data)}")
print(f"Slices w/ infection: {total_inf_slices} ({100*total_inf_slices/len(All_data):.1f}%)")

dataset = (All_data, All_lung, All_inf)
torch.save(dataset, OUTPUT_PT, _use_new_zipfile_serialization=False)
print(f"Saved -> {OUTPUT_PT}")
