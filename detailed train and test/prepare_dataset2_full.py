# -*- coding: utf-8 -*-
"""
prepare_dataset2_full.py

Merge Training_data2a.pt and Validation_data2a.pt into Dataset2_full.pt
for cross-dataset evaluation.
"""

import os
import torch

def merge_dataset2():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_path = os.path.join(base_dir, "Training_data2a.pt")
    val_path = os.path.join(base_dir, "Validation_data2a.pt")
    out_path = os.path.join(base_dir, "Dataset2_full.pt")
    
    print("Loading Training data...")
    tr_data, tr_y, tr_y2 = torch.load(train_path, weights_only=False)
    
    print("Loading Validation data...")
    vl_data, vl_y, vl_y2 = torch.load(val_path, weights_only=False)
    
    print("Merging...")
    full_data = tr_data + vl_data
    full_y = tr_y + vl_y
    full_y2 = tr_y2 + vl_y2
    
    print(f"Total slices: {len(full_data)}")
    
    print(f"Saving to {out_path}...")
    torch.save((full_data, full_y, full_y2), out_path, _use_new_zipfile_serialization=False)
    print("Done!")

if __name__ == '__main__':
    merge_dataset2()
