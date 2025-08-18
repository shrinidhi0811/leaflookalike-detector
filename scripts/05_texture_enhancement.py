import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from skimage.filters import gabor, unsharp_mask
from skimage import img_as_ubyte


# Leaf class folder names
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


# Base directories
base_input_dir = r"D:\leaflookalike-detector\dataset_bg_removed\train"
base_output_dir = r"D:\leaflookalike-detector\dataset_texture_enhanced\train"


# LBP and Gabor Parameters
radius = 2
n_points = 8 * radius
method = "uniform"

frequencies = [0.2, 0.3]
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

#
# Image Processing Function
def process_image(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    unsharp = unsharp_mask(gray, radius=1, amount=1)
    unsharp = img_as_ubyte(unsharp)

    lbp = local_binary_pattern(unsharp, n_points, radius, method).astype(np.uint8)

    gabor_responses = []
    for theta in orientations:
        for freq in frequencies:
            real, imag = gabor(unsharp, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)
            gabor_responses.append(mag)
    gabor_combined = np.max(np.array(gabor_responses), axis=0)
    gabor_combined = img_as_ubyte(gabor_combined / gabor_combined.max())

    merged = cv2.merge([
        cv2.equalizeHist(lbp),
        gabor_combined,
        unsharp
    ])

    filename = os.path.basename(image_path)
    output_path = os.path.join(save_path, filename)
    cv2.imwrite(output_path, merged)


# Process All Folders
for folder_name in folder_names:
    input_folder = os.path.join(base_input_dir, folder_name)
    output_folder = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, file)
            process_image(img_path, output_folder)

    print(f"Finished processing: {folder_name}")

print("Texture enhancement completed!")
