import os
import shutil
import random
from tqdm import tqdm

# Paths
DATASET_PATH = "Preprocessed_Dataset"
OUTPUT_PATH = "Dataset_Split"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set seed for reproducibility
random.seed(42)

def split_dataset():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for plant_type in os.listdir(DATASET_PATH):
        plant_path = os.path.join(DATASET_PATH, plant_type)
        if not os.path.isdir(plant_path):
            continue

        for condition in os.listdir(plant_path):
            condition_path = os.path.join(plant_path, condition)
            if not os.path.isdir(condition_path):
                continue

            # Get all images
            images = os.listdir(condition_path)
            random.shuffle(images)

            total = len(images)
            train_end = int(total * TRAIN_RATIO)
            val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:]
            }

            # Copy files into new structure
            for split_name, split_files in splits.items():
                split_dir = os.path.join(OUTPUT_PATH, split_name, plant_type, condition)
                os.makedirs(split_dir, exist_ok=True)

                for img in tqdm(split_files, desc=f"{plant_type}/{condition}/{split_name}"):
                    src_path = os.path.join(condition_path, img)
                    dst_path = os.path.join(split_dir, img)
                    shutil.copy2(src_path, dst_path)

    print("\n✅ Dataset successfully split into Train / Validation / Test sets!")
    print(f"📂 Output folder: {OUTPUT_PATH}")

if __name__ == "__main__":
    split_dataset()
