# src/preview_image.py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "chest_xray"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def show_random(split="train"):
    folder = DATA_DIR / split
    if not folder.exists():
        print("Folder not found:", folder)
        return

    classes = [d.name for d in folder.iterdir() if d.is_dir()]
    if not classes:
        print("No class directories found in", folder)
        return

    chosen_class = random.choice(classes)
    images = list((folder / chosen_class).glob("*"))
    if not images:
        print("No images found in", folder / chosen_class)
        return

    chosen_image = random.choice(images)
    chosen_path = str(chosen_image)
    print("Selected image:", chosen_path)

    img = cv2.imread(chosen_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image (cv2.imread returned None).")
        return

    print("Image shape:", img.shape, "dtype:", img.dtype)

    # Save a copy so you can open it in explorer
    save_path = OUT_DIR / "sample_preview.png"
    cv2.imwrite(str(save_path), img)
    print("Saved preview to:", save_path)

    # Show with matplotlib
    try:
        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap='gray')
        plt.title(f"{split} → {chosen_class} → {chosen_image.name}")
        plt.axis('off')
        plt.show(block=False)   # don't block execution
        plt.pause(2)            # keep figure open briefly
        # plt.show()             # alternative: blocking call (uncomment if you prefer)
    except Exception as e:
        print("Matplotlib show failed:", e)

    # Also try native OpenCV window (works on Windows)
    try:
        cv2.imshow("Preview (press any key to close)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("cv2.imshow failed:", e)

if __name__ == "__main__":
    split = "train"
    if len(sys.argv) > 1:
        split = sys.argv[1]
    show_random(split)