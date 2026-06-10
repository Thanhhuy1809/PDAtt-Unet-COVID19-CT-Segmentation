import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import cv2

# Import Architecture
import Architectures as networks


def visualize_3_samples():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------------
    # CONFIG: Train D3 -> Test D1
    # ------------------------------------------------------------------------
    model_path = os.path.join(base_dir, "Model_dataset3_new", "PYAttUNet", "Models", "0_bt.pt")
    dataset_path = os.path.join(base_dir, "Dataset1_full.pt")

    # Kiem tra xem file co ton tai khong
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return

    print("=> Loading Dataset 1...")
    data, y, y2 = torch.load(dataset_path, weights_only=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    # Khởi tạo mô hình PYAttUNet
    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    segm = nn.Sigmoid()
    
    # Tiền xử lý ảnh (giống lúc test)
    img_size = 224
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Chọn ngẫu nhiên 3 mẫu từ tập dữ liệu
    num_samples = len(data)
    sample_indices = random.sample(range(num_samples), 3)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.01, hspace=0.01) # Xóa khoảng trắng giữa các ảnh
    fig.suptitle('Dataset 1 Prediction (Train D3 -> Test D1)', fontsize=18, fontweight='bold', color='navy')
    
    col_titles = ["Image", "GT Mask", "Prediction", "Overlay (GT=Red, Pred=Blue)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14, fontweight='bold')

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            # 1. Chuẩn bị dữ liệu đầu vào
            img_np = np.array(data[idx]).astype(np.uint8)
            gt_inf = np.array(y2[idx]).astype(np.uint8) # Chỉ lấy nhãn Viêm phổi (Infection)
            
            aug = transform(image=img_np)
            img_tensor = aug["image"].unsqueeze(0).to(device)
            
            # 2. Dự đoán
            outputs_inf, outputs_lung = model(img_tensor)
            pred_inf = (segm(outputs_inf) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            
            # Resize lại CT gốc để hiển thị
            img_disp = cv2.resize(img_np, (img_size, img_size))
            gt_inf_disp = cv2.resize(gt_inf, (img_size, img_size))
            
            # Xóa trục tọa độ cho tất cả các subplot
            for col in range(4):
                axes[row, col].axis('off')

            # Cột 1: Image (CT Gốc)
            axes[row, 0].imshow(img_disp, cmap='gray')
            
            # Cột 2: GT Mask (Chỉ hiển thị trắng đen)
            axes[row, 1].imshow(gt_inf_disp > 0, cmap='gray')
            
            # Cột 3: Pred (>0.5) (Chỉ hiển thị trắng đen)
            axes[row, 2].imshow(pred_inf > 0, cmap='gray')
            
            # Cột 4: Overlay (GT=Red, Pred=Blue)
            # Tạo RGBA cho GT (Đỏ)
            red_rgba = np.zeros((img_size, img_size, 4), dtype=np.float32)
            red_rgba[:, :, 0] = 1.0 # Kênh R
            red_rgba[:, :, 3] = (gt_inf_disp > 0).astype(np.float32) * 0.4 # Kênh Alpha
            
            # Tạo RGBA cho Pred (Xanh lam)
            blue_rgba = np.zeros((img_size, img_size, 4), dtype=np.float32)
            blue_rgba[:, :, 2] = 1.0 # Kênh B
            blue_rgba[:, :, 3] = (pred_inf > 0).astype(np.float32) * 0.4 # Kênh Alpha
            
            # Hiển thị
            axes[row, 3].imshow(img_disp, cmap='gray')
            axes[row, 3].imshow(red_rgba)
            axes[row, 3].imshow(blue_rgba)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(base_dir, "visualization_cross_testing.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Result saved at: {save_path}")
    # plt.show()


if __name__ == '__main__':
    visualize_3_samples()
