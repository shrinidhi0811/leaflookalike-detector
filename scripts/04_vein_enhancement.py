import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import exposure
import tensorflow as tf

def vein_enhancement(image_path):
    # Step 0: Load image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor for TF integration
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32) / 255.0

    # Step 1: Extract green channel (better for veins)
    green_channel = img_rgb[:, :, 1]

    # Step 2: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_channel)

    # Step 3: Apply Frangi filter (multiscale)
    frangi_filtered = frangi(green_clahe, scale_range=(1, 4), scale_step=1)

    # Normalize Frangi output to 0–255
    frangi_norm = cv2.normalize(frangi_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 4: Small morphological top-hat (3–5 px kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    top_hat = cv2.morphologyEx(frangi_norm, cv2.MORPH_TOPHAT, kernel)

    # Step 5: Contrast stretch
    p2, p98 = np.percentile(top_hat, (2, 98))
    contrast_stretched = exposure.rescale_intensity(top_hat, in_range=(p2, p98))

    # Convert final to uint8
    final_result = (contrast_stretched * 255).astype(np.uint8) if contrast_stretched.max() <= 1 else contrast_stretched

    return img_rgb, final_result


image_folder = r"D:\leaflookalike-detector\dataset_augmented_1\train\alpinia_galanga"
output_folder = r"D:\leaflookalike-detector\dataset_vein_enhanced\alphinia_galanga"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    try:
        original, vein_map = vein_enhancement(img_path)
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, vein_map)
    
    except:
        continue
