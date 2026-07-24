import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_DIR = os.environ.get("LEAF_PROJECT_DIR", SCRIPT_DIR)
OUTPUT_DIR = os.environ.get("LEAF_OUTPUT_DIR", PROJECT_DIR)
BASE_DIR = os.environ.get("LEAF_SPLIT_DIR", os.path.join(PROJECT_DIR, "MultiClass_Dataset_Split_New"))
TEST_DIR = os.path.join(BASE_DIR, "test")

MODEL_PATH = os.environ.get(
    "LEAF_MODEL_PATH",
    os.path.join(OUTPUT_DIR, "new_multiclass_resnet50_best.keras"),
)
CLASS_NAMES_PATH = os.path.join(OUTPUT_DIR, "new_multiclass_class_names.json")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "new_multiclass_resnet50_confusion_matrix.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "new_multiclass_classification_report.txt")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_class_names(class_names_path, fallback_dir):
    """Load class names from JSON, or fall back to test directory order."""
    if os.path.exists(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as file:
            return json.load(file)

    return sorted(
        entry for entry in os.listdir(fallback_dir)
        if os.path.isdir(os.path.join(fallback_dir, entry))
    )


# ==========================================
# 3. LOAD TEST DATA
# ==========================================
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Test directory not found. Check BASE_DIR.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project directory: {PROJECT_DIR}")
print(f"Dataset split directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

class_names = load_class_names(CLASS_NAMES_PATH, TEST_DIR)
print(f"Using {len(class_names)} classes for evaluation.")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode="categorical",
    shuffle=False,
)


# ==========================================
# 4. LOAD MODEL AND PREDICT
# ==========================================
print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

print("Running predictions...")
predictions = model.predict(test_generator, verbose=1)
predicted_indices = np.argmax(predictions, axis=1)
true_indices = test_generator.classes

accuracy = np.mean(predicted_indices == true_indices)
cm = confusion_matrix(true_indices, predicted_indices)
per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
gmean = float(np.exp(np.mean(np.log(np.clip(per_class_recall, 1e-12, 1.0)))))


# ==========================================
# 5. REPORT RESULTS
# ==========================================
print("\n" + "=" * 60)
print("MULTICLASS RESNET50 EVALUATION")
print("=" * 60)
print(f"Total test images : {len(true_indices)}")
print(f"Correct predictions: {(predicted_indices == true_indices).sum()}")
print(f"Wrong predictions  : {(predicted_indices != true_indices).sum()}")
print(f"Top-1 accuracy     : {accuracy * 100:.2f}%")
print(f"G-mean (recall)    : {gmean * 100:.2f}%")
print("=" * 60)

report = classification_report(
    true_indices,
    predicted_indices,
    target_names=class_names,
    digits=4,
)

print("\nDetailed classification report:\n")
print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(report)


# ==========================================
# 6. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(22, 18))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Multiclass ResNet50 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

print(f"\nClassification report saved to: {REPORT_PATH}")
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
