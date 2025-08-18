import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import exposure
import tensorflow as tf

def vein_enhancement(image_path):
    # Step 0: Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Extract green channel
    green_channel = img_rgb[:, :, 1]

    # Step 2: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_channel)

    # Step 3: Apply Frangi filter
    frangi_filtered = frangi(green_clahe, scale_range=(1, 4), scale_step=1)

    # Normalize to 0–255
    frangi_norm = cv2.normalize(frangi_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 4: Morphological top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    top_hat = cv2.morphologyEx(frangi_norm, cv2.MORPH_TOPHAT, kernel)

    # Step 5: Contrast stretching
    p2, p98 = np.percentile(top_hat, (2, 98))
    contrast_stretched = exposure.rescale_intensity(top_hat, in_range=(p2, p98))

    # Ensure uint8 format
    final_result = (contrast_stretched * 255).astype(np.uint8) if contrast_stretched.max() <= 1 else contrast_stretched

    return final_result

# Base directories
base_input_dir = r"D:\leaflookalike-detector\dataset_bg_removed\train"
base_output_dir = r"D:\leaflookalike-detector\dataset_vein_enhanced\train"

# Folder names to process
folder_names = [
    "alpinia_galanga",
    "azadirachta_indica",
    "basella_alba",
    "jasminum",
    "mentha",
    "murraya_koenigii",
    "nerium_oleander",
    "plectranthus_amboinicus",
    "syzygium_jambos",
    "trigonella_foenum_graecum"
]

for folder_name in folder_names:
    image_folder = os.path.join(base_input_dir, folder_name)
    output_folder = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        try:
            vein_map = vein_enhancement(img_path)
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, vein_map)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"Finished: {folder_name}")

print("Vein enhancement completed for all folders.")
