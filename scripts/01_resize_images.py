import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import hashlib

# Paths (change if needed)
ROOT = r"D:/leaflookalike-detector"
SRC_TRAIN = os.path.join(ROOT, "dataset", "train")
DST_PREP_TRAIN = os.path.join(ROOT, "dataset_preprocessed", "train")

TARGET_SIZE = (224, 224)   # MobileNetV3 standard

# Create destination dirs
os.makedirs(DST_PREP_TRAIN, exist_ok=True)

# Optional: enable duplicate detection (exact file duplicates)
detect_duplicates = True
seen_hashes = set()

# Logging
bad_files = []
copied_count = {}

def file_hash(path, blocksize=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            h.update(block)
    return h.hexdigest()

for class_name in sorted(os.listdir(SRC_TRAIN)):
    src_class_dir = os.path.join(SRC_TRAIN, class_name)
    if not os.path.isdir(src_class_dir):
        continue
    dst_class_dir = os.path.join(DST_PREP_TRAIN, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)
    copied_count[class_name] = 0

    files = sorted(os.listdir(src_class_dir))
    for fname in tqdm(files, desc=f"Processing {class_name}", unit="img"):
        src_path = os.path.join(src_class_dir, fname)
        # basic filter
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            # skip non-image files
            continue
        try:
            # Optional duplicate check
            if detect_duplicates:
                try:
                    h = file_hash(src_path)
                    if h in seen_hashes:
                        # duplicate - skip
                        continue
                    seen_hashes.add(h)
                except Exception:
                    pass

            # open & convert to RGB
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE, resample=Image.BILINEAR)

                # Save - keep same filename
                dst_path = os.path.join(dst_class_dir, fname)
                im.save(dst_path, format="JPEG", quality=90)
                copied_count[class_name] += 1

        except UnidentifiedImageError:
            bad_files.append(src_path)
        except Exception as e:
            bad_files.append((src_path, str(e)))

# Summary report
print("\n=== Preprocessing Summary ===")
total = 0
for cls, cnt in copied_count.items():
    print(f"{cls}: {cnt} images")
    total += cnt
print(f"Total processed images: {total}")
if bad_files:
    print("\nProblems with the following files (corrupted/unreadable/other):")
    for b in bad_files[:50]:
        print(" ", b)
else:
    print("No corrupt/unreadable images detected.")
