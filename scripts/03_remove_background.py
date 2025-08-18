import os
from rembg import remove
from PIL import Image
from io import BytesIO

# Base input and output directories
base_input_dir = r"D:\leaflookalike-detector\dataset_augmented_1\train"
base_output_dir = r"D:\leaflookalike-detector\dataset_bg_removed\train"

# List of folder names
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
    input_folder = os.path.join(base_input_dir, folder_name)
    output_folder = os.path.join(base_output_dir, folder_name)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the current input folder
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

            # Save result as PNG
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            final_img.convert("RGB").save(output_path, "PNG")

    print(f"Done processing: {folder_name}")

print("Background removed and replaced with black for all folders.")
