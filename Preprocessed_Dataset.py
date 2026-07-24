import os
import cv2
import numpy as np
from tqdm import tqdm
import hashlib

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We use 'r' before the string to handle Windows backslashes correctly
DATASET_PATH = "."

# We will create the clean dataset inside the same folder, clearly marked
OUTPUT_PATH_NAME = "Preprocessed_Dataset"
OUTPUT_PATH = os.path.join(DATASET_PATH, OUTPUT_PATH_NAME)

IMAGE_SIZE = (224, 224) # ResNet50 Standard

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def is_blurry(image, threshold=100.0):
    """Detect if an image is blurry using variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def compute_hash(image):
    """Compute a hash for duplicate detection."""
    image_bytes = cv2.imencode('.png', image)[1].tobytes()
    return hashlib.md5(image_bytes).hexdigest()

def preprocess_images():
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Created output folder: {OUTPUT_PATH}")

    seen_hashes = set()
    total_processed = 0
    total_saved = 0
    blurry_count = 0
    duplicate_count = 0

    print(f"Scanning directory: {DATASET_PATH}...\n")

    # Loop through items in D:\Leaf Disease Detection
    for plant_type in os.listdir(DATASET_PATH):
        plant_path = os.path.join(DATASET_PATH, plant_type)
        
        # SAFETY CHECK: 
        # 1. Ignore the script file itself
        # 2. Ignore the output folder we just created
        if not os.path.isdir(plant_path) or plant_type == OUTPUT_PATH_NAME:
            continue

        # Now we are inside a crop folder (e.g., 'Tomato Leaves')
        # We expect to find 'Disease' and 'Healthy' folders here
        for condition in os.listdir(plant_path):
            condition_path = os.path.join(plant_path, condition)
            
            # Skip if it's not a folder
            if not os.path.isdir(condition_path):
                continue

            # Create the matching structure in the output folder
            # e.g., D:\Leaf Disease Detection\Preprocessed_Dataset\Tomato Leaves\Disease
            output_dir = os.path.join(OUTPUT_PATH, plant_type, condition)
            os.makedirs(output_dir, exist_ok=True)

            print(f"Processing: {plant_type} -> {condition}")

            # Iterate over images
            for img_name in tqdm(os.listdir(condition_path)):
                img_path = os.path.join(condition_path, img_name)

                try:
                    # Read Image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    total_processed += 1

                    # 1. Resize (Crucial for ResNet)
                    img = cv2.resize(img, IMAGE_SIZE)

                    # 2. Check blur
                    if is_blurry(img):
                        blurry_count += 1
                        continue 

                    # 3. Check duplicates
                    img_hash = compute_hash(img)
                    if img_hash in seen_hashes:
                        duplicate_count += 1
                        continue 
                    seen_hashes.add(img_hash)

                    # 4. Save Image (Raw pixels, no normalization)
                    save_path = os.path.join(output_dir, img_name)
                    cv2.imwrite(save_path, img)
                    total_saved += 1

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    # ==========================================
    # 3. SUMMARY
    # ==========================================
    print("\n" + "="*50)
    print("✅ PREPROCESSING COMPLETE")
    print("="*50)
    print(f"📸 Total images scanned   : {total_processed}")
    print(f"💨 Blurry images removed  : {blurry_count}")
    print(f"🧩 Duplicate images removed : {duplicate_count}")
    print(f"✅ Final Clean Dataset    : {total_saved}")
    print("="*50)
    print(f"Your data is ready at:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_images()