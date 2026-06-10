import os
import numpy as np
import cv2
import nibabel as nib
import torch
from sklearn.model_selection import train_test_split
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

database_pathh = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\data\9CTscans\Dataset2'
database_path_rp_im = os.path.join(database_pathh, 'rp_im')

if not os.path.exists(database_path_rp_im):
    print("Dataset directory not found:", database_path_rp_im)
    exit()

Ct_scans = sorted_alphanumeric(os.listdir(database_path_rp_im))
indicies = list(range(len(Ct_scans)))
train_spt, val_spt = train_test_split(indicies, test_size=0.3, random_state=42)

Training_data, Training_lung, Training_inf = [], [], []
Validation_data, Validation_lung, Validation_inf = [], [], []

print("Preparing Dataset...")
for idx, Ct_scan in enumerate(Ct_scans):
    print(f"Processing {Ct_scan}...")
    slice_samples = os.path.join(database_pathh, 'rp_im', Ct_scan)
    mask_samples = os.path.join(database_pathh, 'rp_lung_msk', Ct_scan)
    inf_samples = os.path.join(database_pathh, 'rp_msk', Ct_scan)

    slices = nib.load(slice_samples).get_fdata()
    masks = nib.load(mask_samples).get_fdata()
    infs = nib.load(inf_samples).get_fdata()

    for i in range(infs.shape[2]):
        slice1 = slices[:, :, i]
        slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
        
        mask1 = masks[:, :, i]
        mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
        
        inf1 = infs[:, :, i]
        inf1 = np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))

        img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if idx in train_spt:
            Training_data.append(img)
            Training_lung.append(mask1)
            Training_inf.append(inf1)
        else:
            Validation_data.append(img)
            Validation_lung.append(mask1)
            Validation_inf.append(inf1)

print("Saving train_dataset.pt...")
torch.save((Training_data, Training_lung, Training_inf), 'train_dataset.pt')

print("Saving val_dataset.pt...")
torch.save((Validation_data, Validation_lung, Validation_inf), 'val_dataset.pt')

print("Dataset prepared successfully!")
