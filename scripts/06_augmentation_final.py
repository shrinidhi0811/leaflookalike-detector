import os
import cv2
import albumentations as A
import glob

# Paths
rgb_root = r"D:\leaflookalike-detector\dataset_augmented_1\train"
vein_root = r"D:\leaflookalike-detector\dataset_vein_enhanced\train"
texture_root = r"D:\leaflookalike-detector\dataset_texture_enhanced\train"

output_root = r"D:\leaflookalike-detector\dataset_augmented_final"
output_rgb = os.path.join(output_root, "rgb")
output_vein = os.path.join(output_root, "vein")
output_texture = os.path.join(output_root, "texture")

# Create output dirs
for path in [output_rgb, output_vein, output_texture]:
    os.makedirs(path, exist_ok=True)

# Augmentation pipeline with additional targets
transform = A.Compose(
    [
        A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=0.5),
        A.RandomScale(scale_limit=0.15, interpolation=cv2.INTER_LINEAR, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    ],
    additional_targets={'vein': 'image', 'texture': 'image'}
)

classes = [
    "alpinia_galanga"#, "azadirachta_indica", "basella_alba", "jasminum",
    #"mentha", "murraya_koenigii", "nerium_oleander", "plectranthus_amboinicus",
    #"syzygium_jambos", "trigonella_foenum_graecum"
]

for cls in classes:
    print(f"\nProcessing class: {cls}")

    dirs = {
        'rgb': os.path.join(rgb_root, cls),
        'vein': os.path.join(vein_root, cls),
        'texture': os.path.join(texture_root, cls)
    }
    outs = {
        'rgb': os.path.join(output_rgb, cls),
        'vein': os.path.join(output_vein, cls),
        'texture': os.path.join(output_texture, cls)
    }
    for out_dir in outs.values():
        os.makedirs(out_dir, exist_ok=True)

    processed = 0

    # Get all .jpg files from RGB folder
    rgb_files = glob.glob(os.path.join(dirs['rgb'], "*.jpg"))

    for rgb_path in rgb_files:
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g., alpinia_galanga_0005

        # Corresponding .png files for vein and texture
        vein_path = os.path.join(dirs['vein'], base_name + ".png")
        texture_path = os.path.join(dirs['texture'], base_name + ".png")

        if not (os.path.exists(vein_path) and os.path.exists(texture_path)):
            print(f"Skipping {base_name}: missing vein or texture image.")
            continue

        # Read all images
        rgb_img = cv2.imread(rgb_path)
        vein_img = cv2.imread(vein_path)
        texture_img = cv2.imread(texture_path)

        if rgb_img is None or vein_img is None or texture_img is None:
            print(f"Skipping {base_name}: failed to read one or more images.")
            continue

        # Save originals as .jpg
        for key, img in zip(('rgb', 'vein', 'texture'), (rgb_img, vein_img, texture_img)):
            out_path = os.path.join(outs[key], base_name + ".jpg")
            cv2.imwrite(out_path, img)

        # Apply augmentations twice
        for i in range(1, 3):
            augmented = transform(image=rgb_img, vein=vein_img, texture=texture_img)
    
            aug_key_map = {'rgb': 'image', 'vein': 'vein', 'texture': 'texture'}

            for key in ('rgb', 'vein', 'texture'):
                out_img = augmented[aug_key_map[key]]
                out_file = os.path.join(outs[key], f"{base_name}_aug{i}.jpg")
                cv2.imwrite(out_file, out_img)

        processed += 1

    print(f" Completed: {processed} images processed for class '{cls}'")

print("\nAll classes processed successfully!")
