import hashlib
import os

import cv2
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = r"D:\Leaf Disease Detection\Multiclass dataset"
OUTPUT_PATH = r"D:\Leaf Disease Detection\Preprocessed_Multiclass_Dataset"

IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
BLUR_THRESHOLD = 100.0


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def is_blurry(image, threshold=BLUR_THRESHOLD):
    """Detect whether an image is blurry using variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    return focus_measure < threshold


def compute_hash(image):
    """Create a hash so duplicate images can be skipped."""
    image_bytes = cv2.imencode(".png", image)[1].tobytes()
    return hashlib.md5(image_bytes).hexdigest()


def iter_image_paths(root_dir):
    """Yield image file paths inside a class directory."""
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(VALID_EXTENSIONS):
                yield os.path.join(current_root, file_name)


def preprocess_multiclass_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Input dataset not found: {DATASET_PATH}")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    seen_hashes = set()
    total_processed = 0
    total_saved = 0
    blurry_count = 0
    duplicate_count = 0
    skipped_classes = []

    class_dirs = [
        entry for entry in sorted(os.listdir(DATASET_PATH))
        if os.path.isdir(os.path.join(DATASET_PATH, entry))
    ]

    print(f"Scanning multiclass dataset: {DATASET_PATH}")
    print(f"Found {len(class_dirs)} classes.\n")

    for class_name in class_dirs:
        class_input_dir = os.path.join(DATASET_PATH, class_name)
        class_output_dir = os.path.join(OUTPUT_PATH, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        image_paths = list(iter_image_paths(class_input_dir))
        if not image_paths:
            skipped_classes.append(class_name)
            continue

        print(f"Processing class: {class_name} ({len(image_paths)} images)")

        for image_path in tqdm(image_paths, desc=class_name):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue

                total_processed += 1
                image = cv2.resize(image, IMAGE_SIZE)

                if is_blurry(image):
                    blurry_count += 1
                    continue

                image_hash = compute_hash(image)
                if image_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(image_hash)

                file_name = os.path.basename(image_path)
                save_path = os.path.join(class_output_dir, file_name)
                cv2.imwrite(save_path, image)
                total_saved += 1

            except Exception as error:
                print(f"Error processing {image_path}: {error}")

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Classes scanned           : {len(class_dirs)}")
    print(f"Images scanned            : {total_processed}")
    print(f"Blurry images removed     : {blurry_count}")
    print(f"Duplicate images removed  : {duplicate_count}")
    print(f"Final clean images saved  : {total_saved}")
    if skipped_classes:
        print(f"Classes with no images    : {', '.join(skipped_classes)}")
    print("=" * 50)
    print(f"Clean dataset saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess_multiclass_dataset()
