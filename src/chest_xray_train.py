# src/train_model.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "chest_xray"
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15  # you can change this to 10/20 etc

# ------------- DATA LOADERS -------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    DATA_ROOT / "train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATA_ROOT / "val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,
)

test_gen = test_datagen.flow_from_directory(
    DATA_ROOT / "test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,
)

print("Class indices:", train_gen.class_indices)  # should be {'NORMAL':0, 'PNEUMONIA':1}

# ------------- BUILD MODEL --------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # first stage: freeze base

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # binary output
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ------------- CALLBACKS ----------------
ckpt_path = OUT / "best_model.h5"

checkpoint_cb = ModelCheckpoint(
    ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
early_stop_cb = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# ------------- TRAIN (FROZEN BASE) -------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
)


# ------------- OPTIONAL FINE-TUNING -------------
# Unfreeze last part of the base model to improve accuracy
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # keep most of it frozen

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

ft_history = model.fit(
    train_gen,
    epochs=5,  # small extra fine-tune
    validation_data=val_gen,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
)

# ------------- SAVE FINAL MODEL -------------
final_model_path = OUT / "final_model.h5"
model.save(final_model_path)
print("Final model saved to:", final_model_path)

# ------------- PLOT TRAINING CURVES -------------
def plot_history(h, title, fname):
    plt.figure(figsize=(10, 4))
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()
    # acc
    plt.subplot(1, 2, 2)
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.suptitle(title)
    out_path = OUT / fname
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


plot_history(history, "Head Training", "history_head.png")
plot_history(ft_history, "Fine-tune Training", "history_finetune.png")

# ------------- EVALUATE ON TEST SET -------------
test_gen.reset()
y_true = test_gen.classes
y_prob = model.predict(test_gen).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("\nClassification report on TEST:")
print(
    classification_report(
        y_true, y_pred, target_names=list(test_gen.class_indices.keys())
    )
)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=list(test_gen.class_indices.keys()),
    yticklabels=list(test_gen.class_indices.keys()),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test)")
cm_path = OUT / "confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print("Saved:", cm_path)

# ROC
try:
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    roc_path = OUT / "roc.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print("Saved:", roc_path)
except Exception as e:
    print("ROC calculation failed:", e)

print("\n✅ Training + evaluation completed.")