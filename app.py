import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# =======================
# CONFIG
# =======================
MODEL_PATH = "trained_model.keras"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Match training params
TARGET_SIZE = (128, 128)
SCALE_INPUTS_0_1 = False  # True if trained with normalization

# =======================
# FLASK APP
# =======================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =======================
# LOAD MODEL
# =======================
model = tf.keras.models.load_model(MODEL_PATH)

# =======================
# LOAD CLASS NAMES FROM VALIDATION DATASET
# =======================

CLASS_NAMES = [
    "Cinammon_RoughBark",
    "Cinammon_StripeCanker",
    "Cinammon__healthy_leaves",
    "Cinammon__leaf_spot_disease",
    "Coffee__Healthy",
    "Coffee__Leaf_rust",
    "Coffee__Miner",
    "Coffee__Phoma",
    "Tea_Healthy",
    "Tea_algal_spot",
    "Tea_gray_blight",
    "Tea_red_leaf_spot"
]


# =======================
# HELPERS
# =======================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_or_path) -> np.ndarray:
    img = Image.open(file_or_path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    if SCALE_INPUTS_0_1:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(file_or_path):
    x = preprocess_image(file_or_path)
    preds = model.predict(x)
    preds = np.array(preds)[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return idx, confidence, preds

# =======================
# ROUTES
# =======================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result_page():
    return render_template("result.html")

@app.route("/history")
def history_page():
    return render_template("history.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save file so it can be shown in results/history
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.seek(0)
    file.save(save_path)

    # Predict
    file.stream.seek(0)
    idx, conf, prob_vec = predict_image(file.stream)

    return jsonify({
        "predicted_index": idx,
        "predicted_class": CLASS_NAMES[idx],
        "confidence": conf,
        "image_url": f"/{save_path}",
        "probabilities": {CLASS_NAMES[i]: float(prob_vec[i]) for i in range(len(CLASS_NAMES))}
    })

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
