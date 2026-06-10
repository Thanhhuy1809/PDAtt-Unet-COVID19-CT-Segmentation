import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
# Chọn model tốt nhất (Run 0 của Dataset 3)
model_path = os.path.join(base_dir, "Model_dataset3_new", "PYAttUNet", "Models", "0_bt.pt")
# Chọn Dataset để test (Đổi sang Validation_data3.pt theo ý bạn)
dataset_path = os.path.join(base_dir, "Validation_data3.pt") 

img_size = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
segm = nn.Sigmoid()

# Load Data
print("Loading data...")
data, y, y2 = torch.load(dataset_path, weights_only=False)

transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# Load Model
print("Loading model...")
model = networks.PYAttUNet()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
model.eval()

# Tìm 10 ảnh ngẫu nhiên có vùng nhiễm đáng kể để vẽ cho đẹp
print("Selecting 10 slices...")
selected_indices = []
np.random.seed(42) # Để kết quả ra cố định
indices = np.random.permutation(len(y))

for i in indices:
    if np.sum(y[i]) > 600: # Chỉ chọn ảnh có vùng nhiễm đủ to
        selected_indices.append(i)
    if len(selected_indices) == 10:
        break

# Custom colormaps để giống hệt trong ảnh mẫu
cmap_lung = ListedColormap(['#F4F8FD', '#0B3B73']) # Nền xanh nhạt, phổi xanh đậm
cmap_inf = ListedColormap(['#FFF5F0', '#7A0101']) # Nền đỏ nhạt, vùng nhiễm đỏ thẫm

fig, axes = plt.subplots(3, 10, figsize=(22, 7))
fig.suptitle('CT Scan Analysis - 10 Slices', fontsize=18, fontweight='bold')

for idx, data_idx in enumerate(selected_indices):
    img_np = np.array(data[data_idx]).astype(np.uint8)
    
    aug = transform(image=img_np)
    img_tensor = aug["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, outputs2 = model(img_tensor)
        # outputs: Infection, outputs2: Lung
        inf_pred = (segm(outputs) > 0.5).squeeze().cpu().numpy()
        lung_pred = (segm(outputs2) > 0.5).squeeze().cpu().numpy()
    
    # Calculate percentage (Infection / Lung Area)
    lung_area = np.sum(lung_pred)
    inf_area = np.sum(inf_pred)
    percentage = (inf_area / lung_area * 100) if lung_area > 0 else 0.0
    
    # --- Row 1: Original ---
    ax_orig = axes[0, idx]
    ax_orig.imshow(img_np, cmap='gray')
    ax_orig.set_title(f'Slice {idx+1}', fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    if idx == 0:
        ax_orig.text(-0.2, 0.5, 'Original', va='center', ha='center', rotation='vertical', transform=ax_orig.transAxes, fontweight='bold', fontsize=14)

    # --- Row 2: Lung Mask ---
    ax_lung = axes[1, idx]
    ax_lung.imshow(lung_pred, cmap=cmap_lung)
    ax_lung.axis('off')
    if idx == 0:
        ax_lung.text(-0.2, 0.5, 'Lung Mask', va='center', ha='center', rotation='vertical', transform=ax_lung.transAxes, fontweight='bold', fontsize=14, color='#0B3B73')

    # --- Row 3: Infection Mask ---
    ax_inf = axes[2, idx]
    ax_inf.imshow(inf_pred, cmap=cmap_inf)
    ax_inf.axis('off')
    ax_inf.set_title(f'{percentage:.1f}%', fontsize=11, fontweight='bold', color='#FF0000')
    if idx == 0:
        ax_inf.text(-0.2, 0.5, 'Infection', va='center', ha='center', rotation='vertical', transform=ax_inf.transAxes, fontweight='bold', fontsize=14, color='#FF0000')

plt.tight_layout()
plt.subplots_adjust(top=0.90, left=0.05, hspace=0.3)
output_path = os.path.join(base_dir, "visualization_result_D3.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved successfully to: {output_path}")
