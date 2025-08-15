import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from shutil import copy2
import random
import cv2

# Paths
TRAIN_DIR = r"D://leaflookalike-detector/dataset_preprocessed/train"
AUG_DIR = r"D://leaflookalike-detector/dataset_augmented_1/train"

# Create augmentation directory
os.makedirs(AUG_DIR, exist_ok=True)

# ----- Augmentation Functions -----
def adjust_brightness_contrast(img):
    """Brightness ±20%, Contrast ±20%"""
    # Brightness
    delta_brightness = random.uniform(-0.2, 0.2)
    img = tf.image.adjust_brightness(img, delta_brightness)
    # Contrast
    contrast_factor = random.uniform(0.8, 1.2)
    img = tf.image.adjust_contrast(img, contrast_factor)
    return img

def adjust_hue_saturation(img):
    """Hue ± ~5-10°, Saturation ±15%"""
    # Hue shift of ~ 5-10 degrees
    delta_hue = random.uniform(-0.03, 0.03)
    img = tf.image.adjust_hue(img, delta_hue)
    # Saturation shift
    sat_factor = random.uniform(0.85, 1.15)
    img = tf.image.adjust_saturation(img, sat_factor)
    return img

def add_noise_blur(img):
    """Gaussian noise (var 0.01-0.02) + mild blur"""
    # Gaussian noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=random.uniform(0.01, 0.02), dtype=tf.float32)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    # Mild blur using average pooling
    k = random.choice([3, 5])
    img = tf.nn.avg_pool2d(tf.expand_dims(img, 0), ksize=k, strides=1, padding='SAME')
    img = tf.squeeze(img, axis=0)
    return img

def random_shadow(img):
    """Natural-looking random shadow"""
    img_np = img.numpy()
    h, w, _ = img_np.shape
    mask = np.ones((h, w), dtype=np.float32)
    # Random polygon points for irregular shadow
    num_points = random.randint(3, 6)
    poly = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(num_points)])
    # Random shadow darkness
    shadow_intensity = random.uniform(0.4, 0.8)
    cv2.fillPoly(mask, [poly], shadow_intensity)
    mask = cv2.GaussianBlur(mask, (random.choice([5, 7]), random.choice([5, 7])), 0)
    mask = np.expand_dims(mask, axis=-1)
    shadow_img = img_np * mask + img_np * (1 - mask) * shadow_intensity
    return tf.convert_to_tensor(shadow_img, dtype=tf.float32)

def random_cutout(img):
    """Random cutout (rect or ellipse) with soft edges"""
    img_np = img.numpy()
    h, w, _ = img_np.shape
    # Cutout size: 5–10% of min dimension
    cutout_size = int(min(h, w) * random.uniform(0.05, 0.1))
    cx = random.randint(0, w - cutout_size)
    cy = random.randint(0, h - cutout_size)
    # Mask initialization
    mask = np.ones((h, w, 3), dtype=np.float32)
    # Random muted color for cutout
    cutout_color = tuple(np.random.uniform(0.3, 0.7, size=3).tolist())
    # 50% chance rectangle, 50% ellipse
    if random.random() < 0.5:
        mask[cy:cy+cutout_size, cx:cx+cutout_size] = cutout_color
    else:
        center = (cx + cutout_size // 2, cy + cutout_size // 2)
        axes = (cutout_size // 2, int(cutout_size * random.uniform(0.4, 0.8)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, cutout_color, -1)
    # Soft edges
    mask = cv2.GaussianBlur(mask, (random.choice([3, 5]), random.choice([3, 5])), 0)
    cutout_img = img_np * mask
    return tf.convert_to_tensor(cutout_img, dtype=tf.float32)

# List of possible augmentations
AUGMENTATIONS = [
    adjust_brightness_contrast,
    adjust_hue_saturation,
    add_noise_blur,
    random_shadow,
    random_cutout
]

# ----- Main Augmentation Loop -----
for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)
    save_path = os.path.join(AUG_DIR, class_name)
    os.makedirs(save_path, exist_ok=True)

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)

        # Skip if not an image
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp")):
            continue

        # Copy original image
        copy2(img_path, save_path)

        # Load and normalize image
        img = load_img(img_path)
        img_array = img_to_array(img) / 255.0

        # Generate 5 variations
        for i in range(5):
            aug_img = tf.convert_to_tensor(img_array, dtype=tf.float32)

            # Random subset & shuffled order
            chosen_augs = random.sample(AUGMENTATIONS, random.randint(3, 4))
            random.shuffle(chosen_augs)

            for aug in chosen_augs:
                aug_img = aug(aug_img)

            aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
            aug_img_pil = array_to_img(aug_img)
            aug_img_pil.save(os.path.join(save_path, f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"))

print("Augmentation completed and saved in:", AUG_DIR)
