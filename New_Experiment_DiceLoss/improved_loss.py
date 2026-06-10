import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

kernel = np.ones((5,5),np.uint8)
TFtotensor = ToTensorV2()

bce_criterion = nn.BCEWithLogitsLoss()

def dice_loss(pred_logits, targets, smooth=1e-5):
    """
    Tính Dice Loss từ Logits. Tự động apply Sigmoid trước khi tính toán.
    """
    pred_probs = torch.sigmoid(pred_logits)
    pred_probs = pred_probs.view(-1)
    targets = targets.view(-1)
    
    intersection = (pred_probs * targets).sum()
    dice = (2. * intersection + smooth) / (pred_probs.sum() + targets.sum() + smooth)
    
    return 1 - dice

def bce_dice_loss(pred_logits, targets):
    """
    Kết hợp 50% BCE và 50% Dice
    """
    bce = bce_criterion(pred_logits, targets)
    dice = dice_loss(pred_logits, targets)
    return 0.5 * bce + 0.5 * dice

def Hybrid_loss_improved(GT_Inf, GT_Lung, Pred_Inf, Pred_Lung):
    """
    Hàm Loss tổng thể chuẩn nhất
    """
    # Giảm trọng số của Edge Loss xuống để không bị bóp nghẹt mô hình
    alpha = 0.1
    
    # 1. Tạo viền (Edge)
    Ed = GT_Inf * 255.0
    Ed = Ed.cpu().numpy().astype(np.uint8)
    Edge_list = []
    
    for i in range(Ed.shape[0]):
        edge_img = cv2.morphologyEx(Ed[i][0], cv2.MORPH_GRADIENT, kernel)
        Edge_list.append(edge_img)
        
    Edge = np.array(Edge_list)
    Edge[Edge > 0.0] = 1.0
    Edge = torch.from_numpy(Edge).float().unsqueeze(1).to(Pred_Inf.device)
    
    # 2. Tính toán các thành phần Loss
    loss_inf = bce_dice_loss(Pred_Inf, GT_Inf)
    loss_lung = bce_dice_loss(Pred_Lung, GT_Lung)
    loss_edge = bce_criterion(Pred_Inf * Edge, Edge)
    
    # 3. Tổng hợp Loss
    loss_total = 0.7 * loss_inf + 0.3 * loss_lung + alpha * loss_edge
    
    return loss_total
