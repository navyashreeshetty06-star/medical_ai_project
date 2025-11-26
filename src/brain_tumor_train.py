# brain_tumor_train.py
# Run from MEDICAL_AI_PROJECT folder:  python src/brain_tumor_train.py

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ------------- PATHS -------------

# folder where this file is
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)

TRAIN_DIR = os.path.join(PROJECT_DIR, "brain_tumor", "Training")
TEST_DIR  = os.path.join(PROJECT_DIR, "brain_tumor", "Testing")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Train dir :", TRAIN_DIR)
print("Test dir  :", TEST_DIR)
print("Output dir:", OUTPUT_DIR)

# ------------- HYPERPARAMETERS -------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Phase 1 and Phase 2 epochs
EPOCHS_FROZEN = 5      # first, train only top layers
EPOCHS_FINE_TUNE = 10  # then, fine-tune base model

# ------------- DATA LOADING -------------

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# save class indices for prediction later
class_indices_path = os.path.join(OUTPUT_DIR, "brain_tumor_class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(class_indices, f, indent=4)
print(f"Saved class indices to {class_indices_path}")

# ------------- MODEL (EfficientNetB0) -------------

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMAGE_SIZE + (3,)
)

# freeze base model for phase 1
base_model.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------- CALLBACKS -------------

best_model_path = os.path.join(OUTPUT_DIR, "brain_tumor_model.h5")

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# ------------- TRAINING PHASE 1 (frozen base) -------------

print("\n========== PHASE 1: Train top layers (base model frozen) ==========\n")
history_frozen = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=val_generator,
    callbacks=callbacks
)

# ------------- TRAINING PHASE 2 (fine-tune last layers) -------------

print("\n========== PHASE 2: Fine-tune base model ==========\n")

# unfreeze last few blocks of EfficientNet
base_model.trainable = True

# you can adjust this if GPU is weak: unfreeze fewer layers
fine_tune_at = len(base_model.layers) - 60
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_generator,
    callbacks=callbacks
)

# ------------- FINAL EVALUATION -------------

print("\n========== FINAL EVALUATION ON TEST SET ==========\n")
test_loss, test_acc = model.evaluate(val_generator)
print(f"Final test accuracy: {test_acc:.4f}")
print(f"Best model saved at: {best_model_path}")