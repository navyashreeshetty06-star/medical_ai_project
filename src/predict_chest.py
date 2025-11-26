# src/predict_image.py
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "chest_model.h5"  # must exist
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # assumes 0: NORMAL, 1: PNEUMONIA


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def load_and_preprocess(img_path: Path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr, img


def predict_image(model, img_path: Path):
    arr, display_img = load_and_preprocess(img_path)
    prob = model.predict(arr)[0][0]  # sigmoid output
    class_idx = int(prob > 0.5)
    label = CLASS_NAMES[class_idx]
    conf = prob if class_idx == 1 else 1 - prob

    print("\n===================================")
    print(f"Image: {img_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.4f}")
    print("===================================\n")

    plt.figure(figsize=(6, 6))
    plt.imshow(display_img)
    plt.title(f"{label} ({conf:.2f})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    model = load_model()
    predict_image(model, img_path)