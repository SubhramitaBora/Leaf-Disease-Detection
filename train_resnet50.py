import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Pointing to your split folders
BASE_DIR = r"D:\Leaf Disease Detection\Dataset_Split"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 

# ==========================================
# 2. HELPER TO SCAN FOLDERS
# ==========================================
# Since your structure is Nested (Crop -> Condition -> Image),
# we need to scan the folders manually to build a DataFrame.

def create_dataframe(root_path):
    filepaths = []
    labels = []
    
    # Check if root path exists
    if not os.path.exists(root_path):
        print(f"Error: Directory not found -> {root_path}")
        return pd.DataFrame()

    # Walk through crop types (Bell Pepper, Citrus, etc.)
    for crop in os.listdir(root_path):
        crop_path = os.path.join(root_path, crop)
        if os.path.isdir(crop_path):
            # Walk through conditions (Healthy/Disease)
            for condition in os.listdir(crop_path):
                condition_path = os.path.join(crop_path, condition)
                if os.path.isdir(condition_path):
                    # Determine Binary Label based on folder name
                    label = None
                    if 'healthy' in condition.lower():
                        label = 'Healthy'
                    elif 'disease' in condition.lower():
                        label = 'Disease'
                    
                    if label:
                        for img in os.listdir(condition_path):
                            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                                filepaths.append(os.path.join(condition_path, img))
                                labels.append(label)
                                
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

# ==========================================
# 3. PREPARE DATA GENERATORS
# ==========================================
print("\nScanning Training Data...")
train_df = create_dataframe(TRAIN_DIR)
print(f"Found {len(train_df)} training images.")

print("Scanning Validation Data...")
val_df = create_dataframe(VAL_DIR)
print(f"Found {len(val_df)} validation images.")

if len(train_df) == 0 or len(val_df) == 0:
    print("ERROR: No images found. Check your BASE_DIR path!")
    exit()

print("\nPreparing Generators...")

# Training: With Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Validation: No Augmentation
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==========================================
# 4. BUILD RESNET50 MODEL
# ==========================================
print("\nBuilding ResNet50 Model...")

# Load Base Model (ImageNet weights, exclude top)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
base_model.trainable = False # Freeze base layers

# Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 5. TRAIN
# ==========================================
print("\nStarting Training...")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_leaf_model.h5', save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 6. SAVE & PLOT
# ==========================================
model.save('resnet50_leaf_disease_final.h5')
print("\nModel saved as 'resnet50_leaf_disease_final.h5'")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('accuracy_plot.png')
print("Plot saved as 'accuracy_plot.png'")