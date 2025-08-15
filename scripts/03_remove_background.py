import os
from rembg import remove
from PIL import Image
from io import BytesIO

# Input and output folder paths
input_folder = r"D:\leaflookalike-detector\dataset_augmented_1\train\alpinia_galanga"     # Folder containing your images
output_folder = r"D:\leaflookalike-detector\dataset_bg_removed\alpinia_galanga"   # Folder to save processed images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        input_path = os.path.join(input_folder, filename)

        # Open image and remove background
        with open(input_path, "rb") as f:
            removed_bg = remove(f.read())

        # Convert bytes back to an image
        img = Image.open(BytesIO(removed_bg)).convert("RGBA")

        # Create a black background
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))

        # Paste the foreground object onto the black background
        final_img = Image.alpha_composite(black_bg, img)

        # Save result
        output_path = os.path.join(output_folder, filename)
        final_img.convert("RGB").save(output_path, "PNG")

print("Background removed and replaced with black for all images.")
