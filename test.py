import torch
import Architectures as networks
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

# 1. Tải model tốt nhất
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = networks.PYAttUNet()  # Hoặc model bạn đã huấn luyện
model.load_state_dict(torch.load(r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\detailed train and test\Models3\PYAttUNet\Models\0_bt.pt', map_location=device))
model.to(device)
model.eval()

# 2. Chuẩn bị transform
img_size = 224
transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


# 3. Hàm dự đoán trên ảnh mới
def predict_single_image(image_path):
    # Đọc và xử lý ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    augmented = transform(image=img)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        pred_infection, pred_lung = model(img_tensor)
        pred_infection = torch.sigmoid(pred_infection) > 0.6
        pred_lung = torch.sigmoid(pred_lung) > 0.6

    # Chuyển về numpy
    pred_infection = pred_infection.squeeze().cpu().numpy().astype(np.uint8) * 255
    pred_lung = pred_lung.squeeze().cpu().numpy().astype(np.uint8) * 255

    return pred_infection, pred_lung


# 4. Sử dụng
infection_mask, lung_mask = predict_single_image(r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\covid-19-pneumonia-ct-abdomen-2 (1).jpg')

# 5. Hiển thị kết quả
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(cv2.imread(r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\covid-19-pneumonia-ct-abdomen-2 (1).jpg'), cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(infection_mask, cmap='gray')
axes[1].set_title('Infection Mask')
axes[2].imshow(lung_mask, cmap='gray')
axes[2].set_title('Lung Mask')
plt.show()