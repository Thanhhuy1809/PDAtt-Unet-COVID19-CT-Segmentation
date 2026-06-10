import torch, numpy as np, cv2, sys, os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

sys.path.insert(0, r'C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2')
import Architectures as networks
from torch.utils.data import Dataset

class SingleDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data, self.y, self.y2 = torch.load(path, weights_only=False)
        self.transform = transform
    def __getitem__(self, index):
        img1 = np.array(self.data[index]).astype(np.uint8)
        y1   = np.array(self.y[index]).astype(np.uint8)
        y2   = np.array(self.y2[index]).astype(np.uint8)
        y1[y1 > 0] = 1
        y2[y2 > 0] = 1
        y33 = (y2 * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
        y3[y3 > 0] = 1
        if self.transform is not None:
            aug = self.transform(image=img1, masks=[y1, y2, y3])
            img1 = aug['image']
            y1   = aug['masks'][0]
            y2   = aug['masks'][1]
            y3   = aug['masks'][2]
        return img1, y2, y1, y3
    def __len__(self): return len(self.data)

val_transforms = A.Compose([A.Resize(224,224), A.Normalize([0,0,0],[1,1,1],255.0), ToTensorV2()])
segm = nn.Sigmoid()

def test_model(model_path, data_path, label):
    ds = SingleDataset(data_path, val_transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=30, shuffle=False)
    model = networks.PYAttUNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    TP=FP=FN=0.0
    with torch.no_grad():
        for x,y,y2,edge in loader:
            out,_ = model(x)
            p = (segm(out)>0.5).squeeze(1).cpu().numpy().astype(int)
            g = (y.float()>0.5).cpu().numpy().astype(int)
            TP+=float(np.sum(((p==1)+(g==1))==2))
            FP+=float(np.sum(((p==1)+(g==0))==2))
            FN+=float(np.sum(((p==0)+(g==1))==2))
    F1=TP/(TP+0.5*(FP+FN)+1e-8)
    print(f'{label}: F1={F1:.4f} TP={TP:.0f} FP={FP:.0f} FN={FN:.0f}')

print('M3 on val2a:')
test_model('Model_dataset3/PYAttUNet/Models/0_bt.pt', 'Validation_data2a.pt', 'M3')
print('M2 on val2a:')
test_model('Model_dataset2/PYAttUNet/Models/0_bt.pt', 'Validation_data2a.pt', 'M2')
