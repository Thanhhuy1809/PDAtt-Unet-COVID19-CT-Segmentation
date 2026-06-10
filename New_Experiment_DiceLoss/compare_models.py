import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Import Architectures từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks

# ========================================
# CẤU HÌNH
# ========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NEW_PATH = 'checkpoints/best_model_ds3_final.pt'
MODEL_OLD_PATH = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Model_dataset3\PYAttUNet\Models\0_bt.pt'
DATA_PATH = 'val_dataset.pt'
OUTPUT_IMAGE = 'comparison_results.png'
IMG_SIZE = 224

# ========================================
# DATA LOADER
# ========================================
class Data_loader_Eval(Dataset):
    def __init__(self, pt_file):
        self.data, self.y, self.y2 = torch.load(pt_file)
    
    def __getitem__(self, index):
        img = np.array(self.data[index]).astype(np.uint8)
        y1 = np.array(self.y[index]).astype(np.uint8) # Lung
        y2 = np.array(self.y2[index]).astype(np.uint8) # Infection
        
        y1[y1 > 0] = 1
        y2[y2 > 0] = 1
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        y1_resized = cv2.resize(y1, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y2_resized = cv2.resize(y2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        img_norm = img_resized / 255.0
        img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1)
        
        return img_tensor, torch.from_numpy(y1_resized).float(), torch.from_numpy(y2_resized).float()

    def __len__(self):
        return len(self.data)

# ========================================
# CHƯƠNG TRÌNH CHÍNH
# ========================================
def main():
    print(f"Đang so sánh trên: {DEVICE}")
    
    # 1. Khởi tạo 2 mô hình
    model_new = networks.PYAttUNet().to(DEVICE)
    model_old = networks.PYAttUNet().to(DEVICE)
    
    # 2. Load trọng số
    model_new.load_state_dict(torch.load(MODEL_NEW_PATH, map_location=DEVICE))
    model_old.load_state_dict(torch.load(MODEL_OLD_PATH, map_location=DEVICE))
    model_new.eval()
    model_old.eval()
    print(">> Đã nạp cả 2 mô hình.")

    # 3. Load dữ liệu
    dataset = Data_loader_Eval(DATA_PATH)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 4. So sánh
    num_samples = 20
    plt.figure(figsize=(20, 2.5 * num_samples))
    
    count = 0
    with torch.no_grad():
        for img, target_lung, target_inf in test_loader:
            if count >= num_samples: break
            
            img_in = img.to(DEVICE)
            
            # Dự đoán model mới
            p_inf_new, p_lung_new = model_new(img_in)
            m_inf_new = (torch.sigmoid(p_inf_new) > 0.5).cpu().numpy()[0][0]
            m_lung_new = (torch.sigmoid(p_lung_new) > 0.5).cpu().numpy()[0][0]
            
            # Dự đoán model cũ
            p_inf_old, p_lung_old = model_old(img_in)
            m_inf_old = (torch.sigmoid(p_inf_old) > 0.5).cpu().numpy()[0][0]
            m_lung_old = (torch.sigmoid(p_lung_old) > 0.5).cpu().numpy()[0][0]
            
            # Chuẩn bị hiển thị
            img_show = img.permute(0, 2, 3, 1).numpy()[0]
            gt_inf = target_inf.numpy()[0]
            gt_lung = target_lung.numpy()[0]
            
            row = count
            imgs = [img_show, gt_lung, gt_inf, m_lung_old, m_inf_old, m_lung_new, m_inf_new]
            titles = ["Ảnh gốc", "GT Phổi", "GT Nhiễm trùng", "CŨ: Phổi", "CŨ: Nhiễm trùng", "MỚI: Phổi", "MỚI: Nhiễm trùng"]
            
            for col in range(7):
                plt.subplot(num_samples, 7, row * 7 + col + 1)
                if col == 0:
                    plt.imshow(imgs[col])
                else:
                    plt.imshow(imgs[col], cmap='jet')
                if row == 0:
                    plt.title(titles[col], fontsize=10)
                plt.axis('off')
            
            count += 1

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f">> Đã xong! Xem kết quả tại: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
