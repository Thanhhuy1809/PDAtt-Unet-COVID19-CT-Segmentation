import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\data\9CTscans\Dataset1"
im_path = f"{base_dir}\\rp_im\\tr_im.nii"
lung_mask_path = f"{base_dir}\\tr_lung_mask\\tr_lungmasks_updated.nii"
lesion_mask_path = f"{base_dir}\\tr_mask.nii\\tr_mask.nii"

# Load data
print("Loading NIfTI files...")
im = nib.load(im_path).get_fdata()
lung = nib.load(lung_mask_path).get_fdata()
lesion = nib.load(lesion_mask_path).get_fdata()

# Find slices where lesion mask is present
lesion_sums = np.sum(lesion, axis=(0, 1))
lesion_slices = np.where(lesion_sums > 0)[0]

# Pick 2 slices that have lesion mask
if len(lesion_slices) >= 2:
    np.random.seed(42) # for reproducibility
    selected_slices = np.random.choice(lesion_slices, 2, replace=False)
else:
    selected_slices = lesion_slices

selected_slices = sorted(selected_slices)

# Plotting: 2 rows (for 2 slices), 3 columns (Image, Lung Mask, Lesion Mask)
fig, axes = plt.subplots(len(selected_slices), 3, figsize=(12, 4 * len(selected_slices)))

# Ensure axes is 2D array even if there is only 1 slice
if len(selected_slices) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, s in enumerate(selected_slices):
    img_slice = im[:, :, s]
    # Rotate 90 degrees to display correctly
    img_slice = np.rot90(img_slice)
    
    lung_slice = np.rot90(lung[:, :, s])
    lesion_slice = np.rot90(lesion[:, :, s])
    
    # 1. CT Image (ảnh gốc)
    axes[i, 0].imshow(img_slice, cmap='gray')
    axes[i, 0].set_title(f" CT Image")
    axes[i, 0].axis('off')
    
    # 2. Lung Mask (chỉ mask phổi)
    axes[i, 1].imshow(lung_slice, cmap='gray')
    axes[i, 1].set_title(f" Lung Mask")
    axes[i, 1].axis('off')
    
    # 3. Lesion Mask (chỉ mask tổn thương)
    axes[i, 2].imshow(lesion_slice, cmap='gray')
    axes[i, 2].set_title(f" Mask")
    axes[i, 2].axis('off')

plt.tight_layout()
output_path = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\sample_visualization.png"
plt.savefig(output_path)
print(f"Saved visualization to {output_path} successfully!")
plt.show()
