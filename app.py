from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ------- MODEL CONFIG --------
# chest_model.h5 is inside the outputs folder
MODEL_PATH = os.path.join("outputs", "chest_model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Pneumonia"]

# load model once when app starts
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(file_storage):
    """Read uploaded file from the form and convert to array for the model."""
    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


# -------- ROUTES --------

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "dashboard.html",
        prediction=None,
        confidence=None
    )


@app.route("/service_dashboard", methods=["GET"])
def service_dashboard():
    return render_template(
        "service_dashboard.html",
        prediction=None,
        confidence=None
    )


# NEW: About Us route
@app.route("/about", methods=["GET"])
def about():
    return render_template("about_us.html")


@app.route("/predict", methods=["POST"])
def predict():
    # "image" must match the name in <input type="file" name="image">
    if "image" not in request.files:
        return redirect(url_for("service_dashboard"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("service_dashboard"))

    img = preprocess_image(file)
    preds = model.predict(img)[0]

    # your prediction logic here...
    if len(preds.shape) == 0 or len(preds) == 1:
        prob_pneu = float(preds if np.isscalar(preds) else preds[0])
        prob_normal = 1 - prob_pneu
        if prob_pneu >= 0.5:
            pred_class = "Pneumonia"
            conf = prob_pneu
        else:
            pred_class = "Normal"
            conf = prob_normal
    else:
        idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[idx]
        conf = float(preds[idx])

    return render_template(
        "service_dashboard.html",
        prediction=pred_class,
        confidence=f"{conf * 100:.2f}%"
    )
if __name__ == "__main__":
    app.run(debug=True)