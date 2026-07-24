import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Point to your TEST folder
TEST_DIR = r"D:\Leaf Disease Detection\Dataset_Split\test"
MODEL_PATH = "resnet50_leaf_disease_final.h5"
IMG_SIZE = (224, 224)
HISTORY_PATH = "binary_training_history.json"

# ==========================================
# 2. LOAD TEST DATA (Handling Nested Folders)
# ==========================================
def create_dataframe(root_path):
    filepaths = []
    labels = []
    
    if not os.path.exists(root_path):
        print(f"Error: Test folder not found at {root_path}")
        return pd.DataFrame()

    print(f"Scanning {root_path}...")
    
    for crop in os.listdir(root_path):
        crop_path = os.path.join(root_path, crop)
        if os.path.isdir(crop_path):
            for condition in os.listdir(crop_path):
                condition_path = os.path.join(crop_path, condition)
                if os.path.isdir(condition_path):
                    # Assign Label: 0 for Disease, 1 for Healthy (Matches training)
                    label = 'Healthy' if 'healthy' in condition.lower() else 'Disease'
                    
                    for img in os.listdir(condition_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            filepaths.append(os.path.join(condition_path, img))
                            labels.append(label)
                            
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

print("Preparing Test Data...")
test_df = create_dataframe(TEST_DIR)
print(f"Found {len(test_df)} test images.")

# Important: Shuffle must be FALSE so we can match predictions to true labels
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=False 
)

# ==========================================
# 3. RUN THE TEST
# ==========================================
print("\nLoading Trained Model...")
model = load_model(MODEL_PATH)

print("Running Predictions... (This may take a minute)")
# Get raw probabilities (0.0 to 1.0)
predictions_prob = model.predict(test_generator, verbose=1)

# Convert to 0 or 1
predictions = (predictions_prob > 0.5).astype(int)

# Get the actual truth from the folders
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) # ['Disease', 'Healthy']

# ==========================================
# 4. SHOW RESULTS
# ==========================================
print("\n" + "="*50)
print("FINAL ACCURACY REPORT")
print("="*50)

# Calculate simple accuracy
correct_count = np.sum(predictions.flatten() == true_labels)
total_count = len(true_labels)
accuracy = correct_count / total_count

print(f"✅ Total Images Tested: {total_count}")
print(f"✅ Correct Predictions: {correct_count}")
print(f"❌ Wrong Predictions  : {total_count - correct_count}")
print(f"🏆 FINAL ACCURACY     : {accuracy * 100:.2f}%")
print("="*50)

print("\nDetailed Report:")
print(classification_report(true_labels, predictions, target_names=class_labels))

# ==========================================
# 5. VISUALIZE MISTAKES (Confusion Matrix)
# ==========================================
print("Generating Confusion Matrix...")
cm = confusion_matrix(true_labels, predictions)

# Calculate G-Mean (geometric mean of class recalls)
with np.errstate(divide="ignore", invalid="ignore"):
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
gmean = float(np.prod(recalls) ** (1.0 / len(recalls))) if len(recalls) else 0.0
print(f"🏅 G-Mean (Recall): {gmean * 100:.2f}%")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.title('Confusion Matrix')
plt.savefig('final_test_results.png')
print("Graph saved as 'final_test_results.png'")

# ==========================================
# 6. OPTIONAL: PLOT TRAINING CURVES
# ==========================================
def plot_training_curves(history_data, output_path="binary_training_curves.png"):
    """Plot training/validation accuracy and loss if history data is available."""
    if not history_data:
        print("No training history provided; skipping training curves.")
        return

    train_acc = history_data.get("accuracy")
    val_acc = history_data.get("val_accuracy")
    train_loss = history_data.get("loss")
    val_loss = history_data.get("val_loss")

    if not all([train_acc, val_acc, train_loss, val_loss]):
        print("Training history missing keys; skipping training curves.")
        return

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training curves saved as '{output_path}'")


if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history_payload = json.load(file)
    plot_training_curves(history_payload)
else:
    print(
        f"No history file found at '{HISTORY_PATH}'. "
        "If you want accuracy/loss curves here, save training history during training."
    )
