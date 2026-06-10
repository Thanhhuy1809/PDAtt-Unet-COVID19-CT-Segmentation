import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import Architectures from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Architectures as networks

# Import our new improved loss
from improved_loss import Hybrid_loss_improved

# ========================================
# CẤU HÌNH LOGGING RA FILE TXT
# ========================================
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("training_log.txt")
sys.stderr = sys.stdout # Bắt luôn cả log lỗi

# ========================================
# DATASET LOADER
# ========================================
class Data_loaderVE(Dataset):
    def __init__(self, pt_file=None, data_tuple=None, transform=None):
        """
        Hỗ trợ load từ file .pt hoặc truyền thẳng tuple (data, y1, y2) vào
        """
        if pt_file is not None:
            self.data, self.y, self.y2 = torch.load(pt_file)
        elif data_tuple is not None:
            self.data, self.y, self.y2 = data_tuple
            
        self.transform = transform

    def __getitem__(self, index):
        img1, y1, y2 = self.data[index], self.y[index], self.y2[index]  # y1: Lung, y2: Infection

        img1 = np.array(img1).astype(np.uint8)
        y1 = np.array(y1).astype(np.uint8)
        y2 = np.array(y2).astype(np.uint8)

        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2])
            image = augmentations["image"]
            mask1 = augmentations["masks"][0]  # lung
            mask2 = augmentations["masks"][1]  # infection

        return image, mask1, mask2

    def __len__(self):
        return len(self.data)

class SubsetDataset(Dataset):
    """
    Wrapper để bọc cái Subset của PyTorch nhằm áp dụng Transform khác nhau cho Train và Val
    khi ta cắt đôi cái Dataset3_full.pt
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        # Lấy dữ liệu gốc từ Data_loaderVE (lúc này chưa bị transform)
        img1, y1, y2 = self.subset.dataset.data[self.subset.indices[index]], \
                       self.subset.dataset.y[self.subset.indices[index]], \
                       self.subset.dataset.y2[self.subset.indices[index]]
                       
        img1 = np.array(img1).astype(np.uint8)
        y1 = np.array(y1).astype(np.uint8)
        y2 = np.array(y2).astype(np.uint8)

        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2])
            image = augmentations["image"]
            mask1 = augmentations["masks"][0]
            mask2 = augmentations["masks"][1]

        return image, mask1, mask2

    def __len__(self):
        return len(self.subset)

# ========================================
# TRANSFORMS
# ========================================
img_size = 224

train_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# ========================================
# HÀM HUẤN LUYỆN LÕI
# ========================================
def train_model(model, train_loader, validate_loader, device, epochs, save_name):
    print(f"\n[BẮT ĐẦU HUẤN LUYỆN] Model sẽ được lưu tại: checkpoints/{save_name}")
    print(f"Tổng số Epochs: {epochs}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    os.makedirs('checkpoints', exist_ok=True)
    best_dice = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 20)

        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            x, y_lung, y_inf = batch
            x = x.to(device)
            y_lung = y_lung.float().unsqueeze(1).to(device)
            y_inf = y_inf.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs_inf, outputs_lung = model(x)
            loss = Hybrid_loss_improved(y_inf, y_lung, outputs_inf, outputs_lung)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        val_acc, val_f1, val_sens, val_spec, val_prec = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(validate_loader, desc="Validation"):
                x, y_lung, y_inf = batch
                x = x.to(device)
                y_lung = y_lung.float().unsqueeze(1).to(device)
                y_inf = y_inf.float().unsqueeze(1).to(device)

                outputs_inf, outputs_lung = model(x)
                loss = Hybrid_loss_improved(y_inf, y_lung, outputs_inf, outputs_lung)
                val_loss += loss.item()

                # Tính Metrics cho Infection Mask
                preds = (torch.sigmoid(outputs_inf) > 0.5).int()
                targets = (y_inf > 0.5).int()

                preds = preds.view(-1)
                targets = targets.view(-1)

                TP = ((preds == 1) & (targets == 1)).sum().float()
                TN = ((preds == 0) & (targets == 0)).sum().float()
                FP = ((preds == 1) & (targets == 0)).sum().float()
                FN = ((preds == 0) & (targets == 1)).sum().float()

                eps = 1e-8
                acc = (TP + TN) / (TP + TN + FP + FN + eps)
                sens = TP / (TP + FN + eps)
                spec = TN / (TN + FP + eps)
                prec = TP / (TP + FP + eps)
                f1 = (2 * prec * sens) / (prec + sens + eps)
                dice = (2 * TP) / (2 * TP + FP + FN + eps)
                iou = TP / (TP + FP + FN + eps)

                val_acc += acc.item()
                val_sens += sens.item()
                val_spec += spec.item()
                val_prec += prec.item()
                val_f1 += f1.item()
                val_dice += dice.item()
                val_iou += iou.item()

        # Tính trung bình các metrics
        num_batches = len(validate_loader)
        val_loss /= num_batches
        avg_acc = (val_acc / num_batches) * 100
        avg_sens = (val_sens / num_batches) * 100
        avg_spec = (val_spec / num_batches) * 100
        avg_prec = (val_prec / num_batches) * 100
        avg_f1 = (val_f1 / num_batches) * 100
        avg_dice = val_dice / num_batches
        avg_iou = val_iou / num_batches

        print(f"\n{'=' * 40}")
        print(f"=> KẾT QUẢ EPOCH {epoch + 1}:")
        print(f"{'=' * 40}")
        print(f"   Train Loss      : {train_loss:.4f}")
        print(f"   Validation Loss : {val_loss:.4f}")
        print(f"----------------------------------------")
        print(f"   Accuracy        : {avg_acc:.2f} %")
        print(f"   F1 Score        : {avg_f1:.2f} %")
        print(f"   Sensitivity     : {avg_sens:.2f} % (Phát hiện vùng bệnh)")
        print(f"   Specificity     : {avg_spec:.2f} % (Nhận diện vùng khỏe)")
        print(f"   Precision       : {avg_prec:.2f} %")
        print(f"----------------------------------------")
        print(f"   Dice Score      : {avg_dice:.4f} (Độ trùng khớp biên)")
        print(f"   IoU Score       : {avg_iou:.4f} (Giao/Hợp)")
        print(f"{'=' * 40}")

        # Kích hoạt Scheduler kiểm tra xem có cần giảm tốc độ LR không
        scheduler.step(avg_dice)

        # Lưu best model theo ĐỘ CHUẨN XÁC Y KHOA (Dice Score)
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f'checkpoints/{save_name}')
            print(f"   [+] Đã lưu Best Model mới! (Dice Score vượt mốc: {best_dice:.4f})")
            
    return model # Trả về model sau khi train xong

# ========================================
# PIPELINE CHÍNH
# ========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = 8
    model = networks.PYAttUNet().to(device)

    # ---------------------------------------------------------
    # PHASE 1: TRAIN TRÊN DATASET 2 (Dữ liệu nhỏ để model khởi động)
    # ---------------------------------------------------------
    print("\n" + "#"*50)
    print("PHASE 1: HUẤN LUYỆN TRÊN DATASET 2")
    print("#"*50)
    if not os.path.exists('train_dataset.pt'):
        print("Dataset 2 không tồn tại! Chạy dataset_prep.py trước.")
        return

    train_set_ds2 = Data_loaderVE(pt_file='train_dataset.pt', transform=train_transform)
    val_set_ds2 = Data_loaderVE(pt_file='val_dataset.pt', transform=val_transforms)

    train_loader_ds2 = DataLoader(train_set_ds2, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_ds2 = DataLoader(val_set_ds2, batch_size=batch_size, shuffle=False)

    # Train 30 Epochs cho Phase 1
    model = train_model(model, train_loader_ds2, val_loader_ds2, device, epochs=30, save_name='best_model_ds2_alpha01.pt')

    # Load lại cái Model tốt nhất của Phase 1 để đi tiếp sang Phase 2
    model.load_state_dict(torch.load('checkpoints/best_model_ds2_alpha01.pt'))
    print(">> Đã nạp thành công Best Model của Dataset 2 để chuẩn bị cho Phase 2.")

    # ---------------------------------------------------------
    # PHASE 2: TRAIN TRÊN DATASET 3 FULL (Dữ liệu khủng để bứt phá)
    # ---------------------------------------------------------
    print("\n" + "#"*50)
    print("PHASE 2: HUẤN LUYỆN CHUYÊN SÂU TRÊN DATASET 3 (FULL)")
    print("#"*50)
    
    dataset3_path = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Dataset3_full.pt'
    
    if not os.path.exists(dataset3_path):
        print(f"Không tìm thấy file: {dataset3_path}")
        return

    # Load file 5.6GB cực lớn (bỏ qua Transform tạm thời để cắt Subset)
    print("Đang nạp Dataset 3 vào RAM. Quá trình này có thể mất vài phút...")
    base_ds3 = Data_loaderVE(pt_file=dataset3_path, transform=None)
    print(f"Nạp thành công! Tổng số ảnh trong Dataset 3: {len(base_ds3)}")
    
    # Chia 80% Train, 20% Val
    train_size = int(0.8 * len(base_ds3))
    val_size = len(base_ds3) - train_size
    train_subset, val_subset = random_split(base_ds3, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Gắn Transform vào 2 Subset
    train_set_ds3 = SubsetDataset(train_subset, transform=train_transform)
    val_set_ds3 = SubsetDataset(val_subset, transform=val_transforms)

    train_loader_ds3 = DataLoader(train_set_ds3, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_ds3 = DataLoader(val_set_ds3, batch_size=batch_size, shuffle=False)

    # Train tiếp 40 Epochs trên Dataset 3
    model = train_model(model, train_loader_ds3, val_loader_ds3, device, epochs=40, save_name='best_model_ds3_final.pt')
    
    print("\n[HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!]")
    print("Kết quả đã được ghi vào file training_log.txt")


if __name__ == "__main__":
    main()
