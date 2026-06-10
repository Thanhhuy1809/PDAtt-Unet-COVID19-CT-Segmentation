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
MODEL_PATH = 'checkpoints/best_model_ds3_final.pt'
DATA_PATH = 'val_dataset.pt'
OUTPUT_IMAGE = 'evaluation_results.png'
IMG_SIZE = 224

# ========================================
# DATA LOADER (Tương tự train.py)
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
        
        # Resize và Normalize đơn giản cho inference
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        y1_resized = cv2.resize(y1, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y2_resized = cv2.resize(y2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        img_norm = img_resized / 255.0
        # Chuyển sang Tensor [C, H, W]
        img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1)
        
        return img_tensor, torch.from_numpy(y1_resized).float(), torch.from_numpy(y2_resized).float()

    def __len__(self):
        return len(self.data)

# ========================================
# CHƯƠNG TRÌNH CHÍNH
# ========================================
def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    
    # 1. Khởi tạo mô hình
    model = networks.PYAttUNet().to(DEVICE)
    
    # 2. Load trọng số
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy file checkpoint tại {MODEL_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(">> Đã nạp mô hình thành công.")

    # 3. Load dữ liệu test
    if not os.path.exists(DATA_PATH):
        print(f"LỖI: Không tìm thấy file dữ liệu tại {DATA_PATH}")
        return
    
    dataset = Data_loader_Eval(DATA_PATH)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f">> Đã nạp tập test ({len(dataset)} ảnh).")

    # 4. Chạy dự đoán và trực quan hóa (lấy 5 mẫu ngẫu nhiên)
    num_samples = 5
    plt.figure(figsize=(15, 3 * num_samples))
    
    count = 0
    with torch.no_grad():
        for i, (img, target_lung, target_inf) in enumerate(test_loader):
            if count >= num_samples: break
            
            img_in = img.to(DEVICE)
            pred_inf_logits, pred_lung_logits = model(img_in)
            
            # Chuyển sang xác suất và mask (threshold 0.5)
            pred_inf = (torch.sigmoid(pred_inf_logits) > 0.5).cpu().numpy()[0][0]
            pred_lung = (torch.sigmoid(pred_lung_logits) > 0.5).cpu().numpy()[0][0]
            
            # Chuẩn bị ảnh để hiển thị
            img_show = img.permute(0, 2, 3, 1).numpy()[0]
            gt_inf = target_inf.numpy()[0]
            gt_lung = target_lung.numpy()[0]
            
            # Vẽ
            row = count
            titles = ["Ảnh CT", "Nhãn Phổi (GT)", "Dự đoán Phổi", "Nhãn Nhiễm trùng (GT)", "Dự đoán Nhiễm trùng"]
            imgs = [img_show, gt_lung, pred_lung, gt_inf, pred_inf]
            
            for col in range(5):
                plt.subplot(num_samples, 5, row * 5 + col + 1)
                if col == 0:
                    plt.imshow(imgs[col])
                else:
                    plt.imshow(imgs[col], cmap='jet')
                if row == 0:
                    plt.title(titles[col])
                plt.axis('off')
            
            count += 1

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f">> Đã lưu ảnh kết quả so sánh tại: {OUTPUT_IMAGE}")
    # plt.show() # Tắt show để chạy script không bị block

if __name__ == "__main__":
    main()
