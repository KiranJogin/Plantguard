import os
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mysqldb import MySQL
import tensorflow as tf
from functools import wraps

# =======================
# CONFIG
# =======================
MODEL_PATH = "trained_model.keras"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
TARGET_SIZE = (128, 128)
SCALE_INPUTS_0_1 = False

# =======================
# FLASK APP
# =======================
app = Flask(__name__)
app.secret_key = "kiranjogin"   # change in production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DB Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'kiranjogin'
app.config['MYSQL_DB'] = 'plantguard'
mysql = MySQL(app)

# =======================
# LOAD MODEL
# =======================
model = tf.keras.models.load_model(MODEL_PATH)

# =======================
# CLASS NAMES
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
# LOAD METADATA
# =======================
metadata_df = pd.read_excel("PlantGuard_Metadata.xlsx")

def get_solution_and_tips(disease, severity, soil, weather):
    row = metadata_df[
        (metadata_df["Disease"] == disease) &
        (metadata_df["Severity"] == severity) &
        (metadata_df["Soil"] == soil) &
        (metadata_df["Weather"] == weather)
    ]
    if not row.empty:
        return row.iloc[0]["Solution"], row.iloc[0]["Tips"]
    return ("No specific solution available. Consult experts.", 
            "Maintain good agricultural practices.")

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

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# =======================
# ROUTES
# =======================
@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/about2")
def about2():
    return render_template("about2.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name, password FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            flash("Signed in successfully!", "success")
            return redirect(url_for("index"))
        flash("Invalid email or password", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        hash_pw = generate_password_hash(password)

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (name, email, password) VALUES (%s,%s,%s)", (name, email, hash_pw))
            mysql.connection.commit()
            flash("Account created successfully! Please sign in.", "success")
            return redirect(url_for("login"))
        except:
            flash("Email already registered", "error")
            return redirect(url_for("register"))
        finally:
            cur.close()

    return render_template("register.html")

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form["email"]
        new_password = request.form["password"]
        confirm = request.form["confirm"]

        if new_password != confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for("forgot"))

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        if user:
            hash_pw = generate_password_hash(new_password)
            cur.execute("UPDATE users SET password=%s WHERE id=%s", (hash_pw, user[0]))
            mysql.connection.commit()
            flash("Password reset successfully! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("No account found with that email", "error")
            return redirect(url_for("forgot"))
    return render_template("forgot.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("home_page"))

# =======================
# PROTECTED ROUTES
# =======================
@app.route("/index")
@login_required
def index():
    soil_types = metadata_df["Soil"].dropna().unique().tolist()
    weather_types = metadata_df["Weather"].dropna().unique().tolist()
    return render_template("index.html", soil_types=soil_types, weather_types=weather_types)

@app.route("/result")
@login_required
def result_page():
    return render_template("result.html")

@app.route("/history")
@login_required
def history_page():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, image_path, predicted_class, confidence, severity, soil, weather, solution, tips, predicted_at FROM predictions ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "image_path": r[1],
            "predicted_class": r[2],
            "confidence": r[3],
            "severity": r[4],
            "soil": r[5],
            "weather": r[6],
            "solution": r[7],
            "tips": r[8],
            "timestamp": r[9].strftime("%Y-%m-%d %H:%M:%S") if r[9] else ""
        })
    return render_template("history.html", history=history)

import sqlite3


@app.route("/delete_prediction/<int:prediction_id>")
@login_required
def delete_prediction(prediction_id):
    cur = mysql.connection.cursor()
    # optional: ensure only the user who owns the prediction can delete it
    cur.execute("DELETE FROM predictions WHERE id = %s", (prediction_id,))
    mysql.connection.commit()
    cur.close()
    flash("Prediction deleted successfully.", "success")
    return redirect(url_for("history_page"))

# =======================
# API: PREDICT IMAGE
# =======================
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    file = request.files.get("image")
    severity = request.form.get("severity")
    soil = request.form.get("soil")
    weather = request.form.get("weather")

    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    idx, conf, prob_vec = predict_image(save_path)
    disease_class = CLASS_NAMES[idx]

    solution, tips = get_solution_and_tips(disease_class, severity, soil, weather)

    cur = mysql.connection.cursor()
    cur.execute("""INSERT INTO predictions
        (image_path, predicted_class, confidence, severity, soil, weather, solution, tips)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (save_path, disease_class, conf, severity, soil, weather, solution, tips))
    mysql.connection.commit()
    cur.close()

    return jsonify({
        "predicted_index": idx,
        "predicted_class": disease_class,
        "confidence": conf,
        "image_url": f"/{save_path}",
        "severity": severity,
        "soil": soil,
        "weather": weather,
        "solution": solution,
        "tips": tips,
        "probabilities": {CLASS_NAMES[i]: float(prob_vec[i]) for i in range(len(CLASS_NAMES))}
    })

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

