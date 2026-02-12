
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import nibabel as nib

# Import models
sys.path.append(r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2')
import Architectures as networks

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


#############################################
# DATA LOADER (giống trong train_test_PDAttUnet.py)
#############################################
class Data_loaderVE(Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train
        self.data, self.y, self.y2 = torch.load(os.path.join(root, train))
        self.transform = transform

    def __getitem__(self, index):
        img1, y1, y2 = self.data[index], self.y[index], self.y2[index]
        img1 = np.array(img1)
        y1 = np.array(y1)
        y2 = np.array(y2)

        if self.transform is not None:
            aug = self.transform(image=img1, mask=y1, mask2=y2)
            img1 = aug['image']
            y1 = aug['mask']
            y2 = aug['mask2']

        return img1, y1.unsqueeze(0), y2.unsqueeze(0), torch.zeros_like(y1.unsqueeze(0))

    def __len__(self):
        return len(self.data)


#############################################
# METRICS CALCULATION
#############################################
def calculate_metrics(preds, targets):
    """Tính toán tất cả metrics"""
    preds = preds.astype(int)
    targets = targets.astype(int)
    
    TP = np.sum(((preds == 1) + (targets == 1)) == 2)
    TN = np.sum(((preds == 0) + (targets == 0)) == 2)
    FP = np.sum(((preds == 1) + (targets == 0)) == 2)
    FN = np.sum(((preds == 0) + (targets == 1)) == 2)
    
    # Dice Score
    dice = (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
    
    # IoU
    IoU = TP / (TP + FP + FN + 1e-8)
    
    # F1 Score
    F1 = TP / (TP + ((1 / 2) * (FP + FN)) + 1e-8)
    
    # Sensitivity (Recall)
    Sens = TP / (TP + FN + 1e-8)
    
    # Specificity
    Spec = 1 - (FP / (FP + TN + 1e-8))
    
    # Precision
    Prec = TP / (TP + FP + 1e-8)
    
    # Accuracy
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-8) * 100
    
    return {
        'Dice': dice,
        'IoU': IoU,
        'F1': F1,
        'Sensitivity': Sens,
        'Specificity': Spec,
        'Precision': Prec,
        'Accuracy': Acc
    }


#############################################
# MODE 1: TEST ẢNH ĐƠN
#############################################
def test_single_image(model_path, image_path, device='cuda'):
    """Test trên 1 ảnh đơn lẻ"""
    print(f"\n{'='*60}")
    print(f"MODE 1: TEST SINGLE IMAGE")
    print(f"{'='*60}")
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transform
    img_size = 224
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    # Load và predict
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    augmented = transform(image=img_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_infection, pred_lung = model(img_tensor)
        pred_infection = torch.sigmoid(pred_infection) > 0.5
        pred_lung = torch.sigmoid(pred_lung) > 0.5
    
    pred_infection = pred_infection.squeeze().cpu().numpy().astype(np.uint8) * 255
    pred_lung = pred_lung.squeeze().cpu().numpy().astype(np.uint8) * 255
    
    # Hiển thị kết quả
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_infection, cmap='jet')
    axes[1].set_title('Infection Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_lung, cmap='jet')
    axes[2].set_title('Lung Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_result_single.png', dpi=150, bbox_inches='tight')
    print(f" Saved: test_result_single.png")
    plt.show()
    
    return pred_infection, pred_lung


#############################################
# MODE 2: TEST TRÊN VALIDATION DATASET
#############################################
def test_on_dataset(model_path, data_path, device='cuda', batch_size=16):
    """Test trên validation dataset với metrics đầy đủ"""
    print(f"\n{'='*60}")
    print(f"MODE 2: TEST ON VALIDATION DATASET")
    print(f"{'='*60}")
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transform
    img_size = 224
    val_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask2': 'mask'})
    
    # Load dataset
    val_dataset = Data_loaderVE(root='./', train=data_path, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Test
    all_preds_infection = []
    all_targets_infection = []
    
    print("Testing...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x, y_infection, y_lung, _ = batch
            x = x.to(device)
            y_infection = y_infection.to(device)
            
            # Predict
            pred_infection, pred_lung = model(x)
            pred_infection = torch.sigmoid(pred_infection) > 0.5
            
            # Convert to numpy
            preds = pred_infection.squeeze(dim=1).cpu().numpy().astype(int)
            targets = (y_infection > 0.5).squeeze(dim=1).cpu().numpy().astype(int)
            
            all_preds_infection.extend(preds)
            all_targets_infection.extend(targets)
    
    # Calculate metrics cho toàn bộ dataset
    all_preds = np.array(all_preds_infection)
    all_targets = np.array(all_targets_infection)
    
    metrics = calculate_metrics(all_preds.flatten(), all_targets.flatten())
    
    # In kết quả
    print(f"\n{'='*60}")
    print(f"METRICS ON VALIDATION DATASET:")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.6f}")
    print(f"{'='*60}")
    
    # Visualize một vài samples
    visualize_predictions(val_loader, model, device, num_samples=5)
    
    return metrics


def visualize_predictions(dataloader, model, device, num_samples=5):
    """Visualize predictions trên một vài samples"""
    model.eval()
    
    # Lấy 1 batch
    x, y_infection, y_lung, _ = next(iter(dataloader))
    x = x.to(device)
    
    with torch.no_grad():
        pred_infection, pred_lung = model(x)
        pred_infection = torch.sigmoid(pred_infection) > 0.5
        pred_lung = torch.sigmoid(pred_lung) > 0.5
    
    # Convert to numpy
    imgs = x.cpu().numpy()
    preds_infection = pred_infection.cpu().numpy()
    targets_infection = y_infection.cpu().numpy()
    preds_lung = pred_lung.cpu().numpy()
    targets_lung = y_lung.cpu().numpy()
    
    # Plot
    num_samples = min(num_samples, len(imgs))
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    for i in range(num_samples):
        # Denormalize image
        img = imgs[i].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(targets_infection[i, 0], cmap='gray')
        axes[i, 1].set_title('GT Infection')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preds_infection[i, 0], cmap='gray')
        axes[i, 2].set_title('Pred Infection')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(targets_lung[i, 0], cmap='gray')
        axes[i, 3].set_title('GT Lung')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(preds_lung[i, 0], cmap='gray')
        axes[i, 4].set_title('Pred Lung')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_result_dataset.png', dpi=150, bbox_inches='tight')
    print(f" Saved: test_result_dataset.png")
    plt.show()


#############################################
# MODE 3: ENSEMBLE TESTING
#############################################
def test_ensemble(model_paths, data_path, device='cuda', batch_size=16):
    """Test với ensemble của nhiều models"""
    print(f"\n{'='*60}")
    print(f"MODE 3: ENSEMBLE TESTING ({len(model_paths)} models)")
    print(f"{'='*60}")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load all models
    models = []
    for i, model_path in enumerate(model_paths):
        model = networks.PYAttUNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f" Loaded model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
    
    # Transform
    img_size = 224
    val_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask2': 'mask'})
    
    # Load dataset
    val_dataset = Data_loaderVE(root='./', train=data_path, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Test
    all_preds_infection = []
    all_targets_infection = []
    
    print("\nTesting with ensemble...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x, y_infection, y_lung, _ = batch
            x = x.to(device)
            y_infection = y_infection.to(device)
            
            # Ensemble prediction (average)
            ensemble_pred = []
            for model in models:
                pred_infection, pred_lung = model(x)
                pred_infection = torch.sigmoid(pred_infection)
                ensemble_pred.append(pred_infection)
            
            # Average predictions
            avg_pred = torch.stack(ensemble_pred).mean(dim=0) > 0.5
            
            # Convert to numpy
            preds = avg_pred.squeeze(dim=1).cpu().numpy().astype(int)
            targets = (y_infection > 0.5).squeeze(dim=1).cpu().numpy().astype(int)
            
            all_preds_infection.extend(preds)
            all_targets_infection.extend(targets)
    
    # Calculate metrics
    all_preds = np.array(all_preds_infection)
    all_targets = np.array(all_targets_infection)
    
    metrics = calculate_metrics(all_preds.flatten(), all_targets.flatten())
    
    # In kết quả
    print(f"\n{'='*60}")
    print(f"ENSEMBLE METRICS ({len(model_paths)} models):")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.6f}")
    print(f"{'='*60}")
    
    return metrics


#############################################
# MODE 4: TEST FILE .NII (10 SLICES) - MỚI 
#############################################
def test_nii_file(model_path, nii_file_path, device='cuda', save_results=True):
    """
    Test file .nii với 10 slices CT scan
    
    Args:
        model_path: đường dẫn model .pt
        nii_file_path: đường dẫn file val_im.nii
        device: 'cuda' hoặc 'cpu'
        save_results: có lưu kết quả không
    """
    print(f"\n{'='*60}")
    print(f"MODE 4: TEST .NII FILE (10 SLICES)")
    print(f"{'='*60}")
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" Model loaded: {os.path.basename(model_path)}")
    
    # Load .nii file
    print(f" Loading .nii file: {nii_file_path}")
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()
    print(f" NIfTI shape: {nii_data.shape}")
    
    # Extract slices (giả sử slices theo axis 2)
    num_slices = min(10, nii_data.shape[2])
    print(f" Testing {num_slices} slices...")
    
    # Transform
    img_size = 224
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    # Test từng slice
    results = []
    
    for slice_idx in tqdm(range(num_slices), desc="Processing slices"):
        # Lấy 1 slice
        slice_img = nii_data[:, :, slice_idx]
        
        # Normalize về 0-255
        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Convert to RGB (model expects 3 channels)
        slice_rgb = np.stack([slice_img, slice_img, slice_img], axis=-1)
        
        # Transform
        augmented = transform(image=slice_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred_infection, pred_lung = model(img_tensor)
            pred_infection = torch.sigmoid(pred_infection) > 0.6
            pred_lung = torch.sigmoid(pred_lung) > 0.6
        
        # Convert to numpy
        pred_infection_np = pred_infection.squeeze().cpu().numpy().astype(np.uint8) * 255
        pred_lung_np = pred_lung.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        results.append({
            'slice_idx': slice_idx,
            'original': slice_img,
            'pred_infection': pred_infection_np,
            'pred_lung': pred_lung_np
        })
    
    # Visualize - Simple & Clear
    print(f"\nVisualizing results...")
    
    fig, axes = plt.subplots(3, num_slices, figsize=(22, 9))
    fig.suptitle('CT Scan Analysis - 10 Slices', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results):
        # Tính infection percentage - ĐÚNG: Chỉ tính infection TRONG phổi
        lung_mask = (result['pred_lung'] > 127).astype(np.uint8)
        infection_mask = (result['pred_infection'] > 127).astype(np.uint8)
        
        # Infection CHỈ trong vùng phổi
        infection_in_lung = infection_mask * lung_mask
        
        lung_pixels = np.sum(lung_mask)
        infection_pixels = np.sum(infection_in_lung)
        infection_pct = (infection_pixels / (lung_pixels + 1e-8)) * 100
        
        # Row 1: Original
        axes[0, i].imshow(result['original'], cmap='gray')
        axes[0, i].set_title(f'Slice {i+1}', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Lung Mask
        axes[1, i].imshow(result['pred_lung'], cmap='Blues', vmin=0, vmax=255)
        axes[1, i].axis('off')
        
        # Row 3: Infection Mask (chỉ trong phổi)
        infection_display = infection_in_lung * 255
        axes[2, i].imshow(infection_display, cmap='Reds', vmin=0, vmax=255)
        axes[2, i].set_title(f'{infection_pct:.1f}%', fontsize=10, color='red', fontweight='bold')
        axes[2, i].axis('off')
    
    # Labels cho từng hàng
    axes[0, 0].text(-0.15, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    fontsize=13, va='center', ha='right', rotation=90, fontweight='bold')
    axes[1, 0].text(-0.15, 0.5, 'Lung Mask', transform=axes[1, 0].transAxes,
                    fontsize=13, va='center', ha='right', rotation=90, fontweight='bold', color='blue')
    axes[2, 0].text(-0.15, 0.5, 'Infection', transform=axes[2, 0].transAxes,
                    fontsize=13, va='center', ha='right', rotation=90, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    if save_results:
        save_path = 'test_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Statistics
    print(f"\n{'='*50}")
    print("Infection Statistics (% of lung area):")
    print(f"{'='*50}")
    total_infection = 0
    total_lung = 0
    for i, result in enumerate(results):
        lung_mask = (result['pred_lung'] > 127).astype(np.uint8)
        infection_mask = (result['pred_infection'] > 127).astype(np.uint8)
        infection_in_lung = infection_mask * lung_mask
        
        lung_pixels = np.sum(lung_mask)
        infection_pixels = np.sum(infection_in_lung)
        infection_pct = (infection_pixels / (lung_pixels + 1e-8)) * 100
        
        total_infection += infection_pixels
        total_lung += lung_pixels
        
        print(f"  Slice {i+1:2d}: {infection_pct:5.1f}%")
    
    overall_pct = (total_infection / (total_lung + 1e-8)) * 100
    print(f"{'='*50}")
    print(f"  Overall: {overall_pct:5.1f}% (average across all slices)")
    print(f"{'='*50}")
    
    print(f"\n{'='*60}")
    print(f" COMPLETED! Processed {num_slices} slices")
    print(f"{'='*60}")
    
    return results


#############################################
# MAIN - SỬ DỤNG
#############################################
if __name__ == "__main__":
    
    # ==================================================
    # CHỌN MODE TEST (uncomment phần bạn muốn dùng)
    # ==================================================
    
    # ------- MODE 1: TEST ẢNH ĐƠN -------
    """
    model_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Models3\PYAttUNet\Models\0_bt.pt'
    image_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\covid-19-pneumonia-ct-abdomen-2 (1).jpg'
    
    pred_infection, pred_lung = test_single_image(model_path, image_path, device='cuda')
    """
    
    # ------- MODE 2: TEST TRÊN VALIDATION DATASET -------
    """
    model_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Models3\PYAttUNet\Models\0_bt.pt'
    data_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Validation_data2.pt'
    
    metrics = test_on_dataset(model_path, data_path, device='cuda', batch_size=16)
    """
    
    # ------- MODE 3: ENSEMBLE TESTING -------
    """
    # Load tất cả 5 best models
    base_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Models3\PYAttUNet\Models'
    model_paths = [
        os.path.join(base_path, f'{i}_bt.pt') for i in range(5)
    ]
    
    data_path = r'C:Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Validation_data2.pt'
    
    metrics = test_ensemble(model_paths, data_path, device='cuda', batch_size=16)
    """
    
    # ------- MODE 4: TEST FILE .NII (10 SLICES) - ĐÃ ACTIVE -------
    
    # Dùng best model (model 3_bt.pt có F1=72.1%)
    model_path = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Models4\PYAttUNet\Models\4_bt.pt'
    nii_file_path = r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\val_im.nii\val_im.nii'
    
    results = test_nii_file(model_path, nii_file_path, device='cuda', save_results=True)
    
    
    print("\n TESTING COMPLETED!")
