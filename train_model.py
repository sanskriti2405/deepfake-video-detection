import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# ==============================
# 1. PATHS (ABSOLUTE FIX)
# ==============================
DATASET_PATH = r"C:\Users\Acer\Desktop\deepfake detection\faces"
OUTPUT_DIR = r"C:\Users\Acer\Desktop\deepfake detection\outputs\metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 2. NOISE FUNCTION
# ==============================
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)
    img = img + noise
    return np.clip(img, 0, 255)

# ==============================
# 3. DATA GENERATOR
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=add_noise,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ==============================
# 4. MODEL
# ==============================
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 5. CALLBACKS
# ==============================
checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==============================
# 6. TRAINING
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# ==============================
# 7. FINE-TUNING
# ==============================
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

# ==============================
# 8. SAVE MODEL
# ==============================
model.save(os.path.join(OUTPUT_DIR, "final_model.h5"))

# ==============================
# 9. PREDICTIONS
# ==============================
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int)

# ==============================
# 10. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ==============================
# 11. ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--')
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ==============================
# 12. CLASSIFICATION REPORT
# ==============================
report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\n📊 Classification Report:\n")
print(report)

# ==============================
# 13. TRAINING GRAPHS
# ==============================
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.close()

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
plt.close()

print(f"\n✅ Everything saved in: {OUTPUT_DIR}")
print(f"🔥 ROC AUC: {roc_auc:.4f}")