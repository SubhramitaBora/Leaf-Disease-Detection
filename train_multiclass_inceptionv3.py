import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_DIR = os.environ.get("LEAF_PROJECT_DIR", SCRIPT_DIR)
OUTPUT_DIR = os.environ.get("LEAF_OUTPUT_DIR", PROJECT_DIR)
BASE_DIR = os.environ.get("LEAF_SPLIT_DIR", os.path.join(PROJECT_DIR, "MultiClass_New_Dataset_Split"))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# InceptionV3 standard input size
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 8
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_TRAINABLE_LAYERS = 15
LABEL_SMOOTHING = 0.05
MAX_CLASS_WEIGHT = 15.0
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "multiclass_inceptionv3_best.keras")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "multiclass_inceptionv3_final.keras")
CLASS_NAMES_PATH = os.path.join(OUTPUT_DIR, "multiclass_inceptionv3_class_names.json")
PLOT_PATH = os.path.join(OUTPUT_DIR, "multiclass_inceptionv3_training_plot.png")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_non_empty_classes(root_dir):
    """Return sorted class folders that contain at least one image."""
    class_names = []

    for entry in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(class_dir):
            continue

        image_count = sum(
            1 for file_name in os.listdir(class_dir)
            if file_name.lower().endswith(VALID_EXTENSIONS)
        )
        if image_count > 0:
            class_names.append(entry)

    return class_names


def count_images_per_class(root_dir, class_names):
    """Count images in each class folder."""
    counts = {}

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        counts[class_name] = sum(
            1 for file_name in os.listdir(class_dir)
            if file_name.lower().endswith(VALID_EXTENSIONS)
        )

    return counts


def compute_class_weights(class_counts, class_indices):
    """Compute inverse-frequency class weights for imbalanced multiclass training."""
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)

    return {
        class_indices[class_name]: min(
            total_images / (num_classes * count),
            MAX_CLASS_WEIGHT,
        )
        for class_name, count in class_counts.items()
        if count > 0
    }


def save_class_names(class_names, output_path):
    """Save class order so evaluation uses the same mapping."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def merge_histories(*histories):
    """Merge multiple Keras History objects into one dict for plotting."""
    merged = {}

    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)

    return merged


def plot_training_curves(history_data, output_path, fine_tune_start_epoch):
    """Save accuracy and loss curves from both frozen and fine-tuning stages."""
    epochs = range(1, len(history_data["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history_data["val_accuracy"], label="Val Accuracy")
    if fine_tune_start_epoch > 0:
        plt.axvline(fine_tune_start_epoch + 0.5, color="gray", linestyle="--", label="Fine-tuning start")
    plt.title("InceptionV3 Multiclass Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data["loss"], label="Train Loss")
    plt.plot(epochs, history_data["val_loss"], label="Val Loss")
    if fine_tune_start_epoch > 0:
        plt.axvline(fine_tune_start_epoch + 0.5, color="gray", linestyle="--", label="Fine-tuning start")
    plt.title("InceptionV3 Multiclass Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_callbacks(best_model_path):
    """Create callbacks focused on preventing overfitting and saving the best model."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            mode="max",
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-6,
        ),
    ]


# ==========================================
# 3. PREPARE DATA
# ==========================================
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("Train/val directories not found. Check BASE_DIR.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project directory: {PROJECT_DIR}")
print(f"Dataset split directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

class_names = get_non_empty_classes(TRAIN_DIR)
if not class_names:
    raise ValueError("No non-empty class folders found in the training directory.")

save_class_names(class_names, CLASS_NAMES_PATH)

print(f"Detected {len(class_names)} classes.")
print("First 10 classes:", class_names[:10])

train_counts = count_images_per_class(TRAIN_DIR, class_names)
print("\nTraining image counts per class:")
for class_name, count in train_counts.items():
    print(f"{class_name}: {count}")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.15,
    brightness_range=(0.85, 1.15),
    channel_shift_range=15.0,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode="categorical",
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode="categorical",
    shuffle=False,
)

class_weights = compute_class_weights(train_counts, train_generator.class_indices)
print("\nClass weights:")
for class_index, weight in sorted(class_weights.items()):
    print(f"{class_index}: {weight:.4f}")


# ==========================================
# 4. BUILD MODEL
# ==========================================
print("\nBuilding InceptionV3 multiclass model...")

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)


# ==========================================
# 5. TRAIN FROZEN BACKBONE
# ==========================================
print("\nStarting stage 1 training (frozen backbone)...")

stage_1_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=build_callbacks(BEST_MODEL_PATH),
    class_weight=class_weights,
)

stage_1_epochs_completed = len(stage_1_history.history.get("accuracy", []))


# ==========================================
# 6. FINE-TUNE TOP INCEPTIONV3 LAYERS
# ==========================================
print("\nStarting stage 2 fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_TRAINABLE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

stage_2_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=stage_1_epochs_completed + FINE_TUNE_EPOCHS,
    initial_epoch=stage_1_epochs_completed,
    callbacks=build_callbacks(BEST_MODEL_PATH),
    class_weight=class_weights,
)


# ==========================================
# 7. SAVE OUTPUTS
# ==========================================
model.save(FINAL_MODEL_PATH)
merged_history = merge_histories(stage_1_history, stage_2_history)
plot_training_curves(
    merged_history,
    PLOT_PATH,
    fine_tune_start_epoch=stage_1_epochs_completed,
)

best_val_accuracy = max(merged_history.get("val_accuracy", [0]))
best_val_loss = min(merged_history.get("val_loss", [0]))

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Best validation loss    : {best_val_loss:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
print(f"Class names saved to: {CLASS_NAMES_PATH}")
print(f"Training plot saved to: {PLOT_PATH}")
