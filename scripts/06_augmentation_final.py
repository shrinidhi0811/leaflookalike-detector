import os
import cv2
import albumentations as A

# Paths
rgb_root = r"D:\leaflookalike-detector\dataset_augmented_1\train"
vein_root = r"D:\leaflookalike-detector\dataset_vein_enhanced\train"
texture_root = r"D:\leaflookalike-detector\dataset_texture_enhanced\train"

output_rgb_root = r"D:\leaflookalike-detector\dataset_augmented_final\rgb"
output_vein_root = r"D:\leaflookalike-detector\dataset_augmented_final\vein"
output_texture_root = r"D:\leaflookalike-detector\dataset_augmented_final\texture"

# Create output dirs if not exist
for root in [output_rgb_root, output_vein_root, output_texture_root]:
    os.makedirs(root, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.Flip(p=0.5),
    A.RandomScale(scale_limit=0.15, interpolation=cv2.INTER_LINEAR, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.7)
])

# Iterate over folders
classes = [
    "alpinia_galanga", "azadirachta_indica", "basella_alba", "jasminum", 
    "mentha", "murraya_koenigii", "nerium_oleander", "plectranthus_amboinicus", 
    "syzygium_jambos", "trigonella_foenum_graecum"
]

for cls in classes:
    print(f"Processing leaf class: {cls} ...")

    rgb_dir = os.path.join(rgb_root, cls)
    vein_dir = os.path.join(vein_root, cls)
    texture_dir = os.path.join(texture_root, cls)

    out_rgb_dir = os.path.join(output_rgb_root, cls)
    out_vein_dir = os.path.join(output_vein_root, cls)
    out_texture_dir = os.path.join(output_texture_root, cls)

    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_vein_dir, exist_ok=True)
    os.makedirs(out_texture_dir, exist_ok=True)

    # Iterate over images
    for fname in os.listdir(rgb_dir):
        rgb_path = os.path.join(rgb_dir, fname)
        vein_path = os.path.join(vein_dir, fname)
        texture_path = os.path.join(texture_dir, fname)

        if not os.path.exists(vein_path) or not os.path.exists(texture_path):
            continue  # skip mismatches

        # Read images
        rgb_img = cv2.imread(rgb_path)
        vein_img = cv2.imread(vein_path)
        texture_img = cv2.imread(texture_path)

        base, ext = os.path.splitext(fname)

        # Save original images as-is
        cv2.imwrite(os.path.join(out_rgb_dir, fname), rgb_img)
        cv2.imwrite(os.path.join(out_vein_dir, fname), vein_img)
        cv2.imwrite(os.path.join(out_texture_dir, fname), texture_img)

        # Apply augmentation twice
        for i in range(1, 3):  # 1 and 2
            augmented = transform(image=rgb_img, masks=[vein_img, texture_img])
            aug_rgb = augmented["image"]
            aug_vein, aug_texture = augmented["masks"]

            # Save augmented images with _aug1 and _aug2 suffix
            cv2.imwrite(os.path.join(out_rgb_dir, f"{base}_aug{i}{ext}"), aug_rgb)
            cv2.imwrite(os.path.join(out_vein_dir, f"{base}_aug{i}{ext}"), aug_vein)
            cv2.imwrite(os.path.join(out_texture_dir, f"{base}_aug{i}{ext}"), aug_texture)

print("Original and 2 augmentations per image saved successfully!")
