import matplotlib.pyplot as plt

# ==========================================
# DATA RECOVERED FROM YOUR SCREENSHOT
# ==========================================
epochs = range(1, 11)

# Training Data (Accuracy & Loss)
train_acc =  [0.7471, 0.9040, 0.9345, 0.9491, 0.9529, 0.9604, 0.9631, 0.9672, 0.9683, 0.9695]
train_loss = [0.7013, 0.2959, 0.2119, 0.1669, 0.1459, 0.1273, 0.1165, 0.1045, 0.0981, 0.0925]

# Validation Data (Accuracy & Loss)
val_acc =  [0.9097, 0.9529, 0.9626, 0.9670, 0.9709, 0.9738, 0.9762, 0.9765, 0.9812, 0.9806]
val_loss = [0.3046, 0.1893, 0.1442, 0.1182, 0.1016, 0.0916, 0.0821, 0.0775, 0.0688, 0.0665]

# ==========================================
# PLOTTING
# ==========================================
plt.figure(figsize=(12, 5))

# 1. Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# 2. Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('MobileNetV2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Save it!
plt.tight_layout()
plt.savefig('mobilenetv2_training_plot.png')
print("✅ Graph saved as 'mobilenetv2_training_plot.png'")
plt.show()