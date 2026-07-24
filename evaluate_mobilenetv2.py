import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CHANGE 1: Use MobileNetV2 Preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_DIR = r"D:\Leaf Disease Detection\Dataset_Split\test"

# CHANGE 2: Point to the new MobileNet model
MODEL_PATH = "mobilenetv2_leaf_disease_final.keras"
IMG_SIZE = (224, 224)

# ==========================================
# 2. LOAD TEST DATA
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
                    label = 'Healthy' if 'healthy' in condition.lower() else 'Disease'
                    
                    for img in os.listdir(condition_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            filepaths.append(os.path.join(condition_path, img))
                            labels.append(label)
                            
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

print("Preparing Test Data...")
test_df = create_dataframe(TEST_DIR)
print(f"Found {len(test_df)} test images.")

# Use the correct preprocessing function
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
print(f"\nLoading Model: {MODEL_PATH}...")
model = load_model(MODEL_PATH)

print("Running Predictions... (This may take a minute)")
predictions_prob = model.predict(test_generator, verbose=1)
predictions = (predictions_prob > 0.5).astype(int)

true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ==========================================
# 4. SHOW RESULTS
# ==========================================
print("\n" + "="*50)
print("FINAL ACCURACY REPORT (MobileNetV2)")
print("="*50)

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
# 5. VISUALIZE MISTAKES
# ==========================================
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.title('MobileNetV2 Confusion Matrix')
plt.savefig('mobilenetv2_confusion_matrix.png')
print("Graph saved as 'mobilenetv2_confusion_matrix.png'")