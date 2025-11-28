import os
import sqlite3
import uuid
import io
import csv
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify, session, redirect,
    url_for, flash, g, send_file
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# ----------------------------------------------------
# OPTIONAL TENSORFLOW (MODEL MAY BE MISSING)
# ----------------------------------------------------
try:
    import tensorflow as tf
except:
    tf = None

# ----------------------------------------------------
# LEAF VALIDATION IMPORTS
# ----------------------------------------------------
from numpy.linalg import norm
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

# Load MobileNetV2 for embeddings
leaf_base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
leaf_embedder = Model(inputs=leaf_base.input, outputs=leaf_base.output)

# Load dataset embeddings
if os.path.exists("dataset_embeddings.npz"):
    DATA = np.load("dataset_embeddings.npz", allow_pickle=True)
    dataset_emb = DATA["emb"]
    dataset_emb_norm = np.linalg.norm(dataset_emb, axis=1)
else:
    dataset_emb = None
    dataset_emb_norm = None
    print("⚠ dataset_embeddings.npz NOT FOUND — leaf validation disabled")

# ----------------------------------------------------
# PDF / QR OPTIONAL
# ----------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
except:
    canvas = None
    colors = None

try:
    import qrcode
except:
    qrcode = None

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
MODEL_PATH = "trained_model.keras"
UPLOAD_FOLDER = "static/uploads"
DB_PATH = "plantguard.db"
METADATA_FILE = "PlantGuard_Metadata_Complete.xlsx"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
TARGET_SIZE = (128, 128)

# ----------------------------------------------------
# FLASK
# ----------------------------------------------------
app = Flask(__name__)
app.secret_key = "kiranjogin"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------
# DATABASE HELPERS
# ----------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db:
        db.close()

def ensure_tables_and_columns():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE, password TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            predicted_class TEXT,
            confidence REAL,
            severity TEXT,
            soil TEXT,
            weather TEXT,
            solution TEXT,
            tips TEXT,
            dosage TEXT,
            biochemical TEXT,
            agronomic TEXT,
            preventive TEXT,
            spray_interval TEXT,
            chemicals TEXT,
            organic TEXT,
            metadata_source TEXT,
            plant_name TEXT,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    try:
        db.execute("ALTER TABLE predictions ADD COLUMN tags TEXT")
    except:
        pass
    db.commit()

# Normalize DB paths
def fix_db_image_paths():
    db = get_db()
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR_ABS = os.path.join(APP_ROOT, "static/uploads")

    rows = db.execute("SELECT id, image_path FROM predictions").fetchall()

    for r in rows:
        old = r["image_path"]
        if not old:
            continue
        if os.path.isabs(old):
            continue

        filename = os.path.basename(old)
        corrected = os.path.join(UPLOAD_DIR_ABS, filename)
        db.execute("UPDATE predictions SET image_path=? WHERE id=?", (corrected, r["id"]))

    db.commit()

# ----------------------------------------------------
# LOAD DISEASE MODEL
# ----------------------------------------------------
model = None
if tf:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✔ Disease model loaded")
    except Exception as e:
        print("⚠ Could not load disease model:", e)
        model = None

# ----------------------------------------------------
# DISEASE CLASSES
# ----------------------------------------------------
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

def plant_from_class(c):
    for p in ["Coffee", "Tea", "Cinammon"]:
        if c.startswith(p):
            return p
    return c.split("_")[0]

# ----------------------------------------------------
# LOAD METADATA
# ----------------------------------------------------
metadata_generic_df = pd.read_excel(METADATA_FILE, sheet_name="Generic_Metadata")
metadata_full_df = pd.read_excel(METADATA_FILE, sheet_name="Full_Metadata_Combinations")

# ----------------------------------------------------
# METADATA LOOKUP
# ----------------------------------------------------
def lookup_metadata(disease, soil, weather):
    disease = disease.strip()
    soil = (soil or "").strip()
    weather = (weather or "").strip()

    # ---- MATCHED ----
    if soil and weather:
        match = metadata_full_df[
            (metadata_full_df["Disease"] == disease) &
            (metadata_full_df["Soil"] == soil) &
            (metadata_full_df["Weather"] == weather)
        ]
        if not match.empty:
            r = match.iloc[0]
            return {
                "severity": r.get("Severity", ""),
                "solution": r.get("Precise Solutions", ""),
                "dosage": r.get("Recommended Dosage", "") or r.get("Dosage", ""),
                "biochemical": r.get("Biochemical Treatment", ""),
                "agronomic": r.get("Agronomic Tips", ""),
                "preventive": r.get("Preventive Measures", ""),
                "spray_interval": r.get("Spray Interval", ""),
                "tips": r.get("Agronomic Tips", ""),
                "chemicals": r.get("Biochemical Treatment", ""),
                "organic": "",
                "metadata_source": "matched"
            }

    # ---- GENERIC ----
    match2 = metadata_generic_df[
        metadata_generic_df["Disease"] == disease
    ]

    if not match2.empty:
        r = match2.iloc[0]
        return {
            "severity": r.get("Severity", ""),
            "solution": r.get("Precise Solutions", ""),
            "dosage": r.get("Dosage", ""),
            "biochemical": r.get("Biochemical Treatment", ""),
            "agronomic": r.get("Agronomic Tips", ""),
            "preventive": r.get("Preventive Measures", ""),
            "spray_interval": r.get("Spray Interval", ""),
            "tips": r.get("Agronomic Tips", ""),
            "chemicals": r.get("Biochemical Treatment", ""),
            "organic": "",
            "metadata_source": "generic"
        }

    # ---- FALLBACK ----
    return {
        "severity": "Unknown",
        "solution": "No data available.",
        "dosage": "",
        "biochemical": "",
        "agronomic": "",
        "preventive": "",
        "spray_interval": "",
        "tips": "",
        "chemicals": "",
        "organic": "",
        "metadata_source": "fallback"
    }
# ----------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(file_path):
    img = Image.open(file_path).convert("RGB").resize(TARGET_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(arr, 0)

def predict_image(file_path):
    if model is None:
        raise RuntimeError("Model not loaded")
    x = preprocess(file_path)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return idx, float(preds[idx]), preds

# ----------------------------------------------------
# LEAF VALIDATION FUNCTION
# ----------------------------------------------------
def is_valid_leaf_image(img_path, threshold=0.55):
    """
    Returns True if uploaded image is similar to dataset embeddings.
    """
    if dataset_emb is None:
        return True  # validation disabled if embeddings missing

    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0).astype(np.float32)

        vec = leaf_embedder.predict(arr, verbose=0)[0]
        vec_norm = norm(vec)

        sims = np.dot(dataset_emb, vec) / (dataset_emb_norm * vec_norm + 1e-12)
        max_sim = float(np.max(sims))

        print("Leaf similarity:", max_sim)
        return max_sim >= threshold

    except Exception as e:
        print("Leaf validation error:", e)
        return False

# ----------------------------------------------------
# AUTH HELPERS
# ----------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about2")
def about2():
    return render_template("about2c.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        pw = request.form["password"]

        db = get_db()
        row = db.execute("SELECT id, password FROM users WHERE email=?", (email,)).fetchone()
        if row and check_password_hash(row["password"], pw):
            session["user_id"] = row["id"]
            return redirect("/index")

        flash("Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users(name,email,password) VALUES(?,?,?)",
                (name, email, password)
            )
            db.commit()
            return redirect("/login")
        except:
            flash("Email already registered")

    return render_template("register.html")

@app.route("/index")
@login_required
def index():
    return render_template("index.html")

@app.route("/result")
@login_required
def result_page():
    return render_template("result.html")

@app.route("/history")
@login_required
def history_page():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY predicted_at DESC",
        (session["user_id"],)
    ).fetchall()
    return render_template("history.html", history=[dict(r) for r in rows])

@app.route("/view/<int:id>")
@login_required
def view(id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE id=? AND user_id=?",
        (id, session["user_id"])
    ).fetchone()

    if not row:
        flash("Not found")
        return redirect("/history")

    rec = dict(row)
    filename = os.path.basename(rec["image_path"])
    rec["image_url"] = f"/static/uploads/{filename}"

    return render_template("view_result.html", prediction=rec)

# ----------------------------------------------------
# DELETE A SINGLE ENTRY
# ----------------------------------------------------
@app.route("/delete_prediction/<int:id>")
@login_required
def delete_prediction(id):
    db = get_db()
    row = db.execute(
        "SELECT image_path FROM predictions WHERE id=? AND user_id=?",
        (id, session["user_id"])
    ).fetchone()

    if row:
        try:
            if os.path.exists(row["image_path"]):
                os.remove(row["image_path"])
        except:
            pass

    db.execute("DELETE FROM predictions WHERE id=? AND user_id=?", (id, session["user_id"]))
    db.commit()

    return redirect("/history")

# ----------------------------------------------------
# DELETE ALL
# ----------------------------------------------------
@app.route("/delete_all_history")
@login_required
def delete_all_history():
    db = get_db()

    rows = db.execute(
        "SELECT image_path FROM predictions WHERE user_id=?",
        (session["user_id"],)
    ).fetchall()

    for r in rows:
        try:
            if os.path.exists(r["image_path"]):
                os.remove(r["image_path"])
        except:
            pass

    db.execute("DELETE FROM predictions WHERE user_id=?", (session["user_id"],))
    db.commit()

    return redirect("/history")

# ----------------------------------------------------
# API: PREDICT
# ----------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    img = request.files.get("image")
    if not img or not allowed(img.filename):
        return jsonify({"error": "Invalid image"}), 400

    soil = request.form.get("soil", "")
    weather = request.form.get("weather", "")
    user_plant = request.form.get("plant", "")

    # -------- SAVE UPLOADED IMAGE --------
    ext = img.filename.rsplit(".", 1)[1].lower()
    unique = f"{uuid.uuid4().hex}.{ext}"

    root = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(root, UPLOAD_FOLDER, unique)
    img.save(abs_path)

    web_url = f"/static/uploads/{unique}"

    # --------- LEAF VALIDATION ----------
    if not is_valid_leaf_image(abs_path):
        try:
            os.remove(abs_path)
        except:
            pass
        return jsonify({
            "error": "This image is NOT a plant leaf. Please upload a clear leaf image."
        }), 400

    # --------- PREDICT DISEASE ----------
    try:
        idx, conf, vec = predict_image(abs_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    disease = CLASS_NAMES[idx]
    plant_name = user_plant if user_plant else plant_from_class(disease)

    # --------- LOOKUP METADATA ----------
    meta = lookup_metadata(disease, soil, weather)

    # --------- SAVE TO DATABASE ----------
    db = get_db()
    db.execute("""
        INSERT INTO predictions(
            user_id, image_path, predicted_class, confidence, severity,
            soil, weather, solution, tips, dosage,
            biochemical, agronomic, preventive, spray_interval,
            chemicals, organic, metadata_source, plant_name, tags
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session["user_id"], abs_path, disease, conf, meta["severity"],
        soil, weather, meta["solution"], meta["tips"], meta["dosage"],
        meta["biochemical"], meta["agronomic"], meta["preventive"], meta["spray_interval"],
        meta["chemicals"], meta["organic"], meta["metadata_source"], plant_name, ""
    ))
    db.commit()

    last_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    # --------- RETURN RESPONSE ----------
    return jsonify({
        "id": last_id,
        "predicted_class": disease,
        "confidence": conf,
        "image_url": web_url,
        "severity": meta["severity"],
        "soil": soil,
        "weather": weather,
        "solution": meta["solution"],
        "tips": meta["tips"],
        "dosage": meta["dosage"],
        "biochemical": meta["biochemical"],
        "agronomic": meta["agronomic"],
        "preventive": meta["preventive"],
        "spray_interval": meta["spray_interval"],
        "chemicals": meta["chemicals"],
        "organic": meta["organic"],
        "metadata_source": meta["metadata_source"],
        "plant_name": plant_name,
        "tags": "",
        "probabilities": {CLASS_NAMES[i]: float(vec[i]) for i in range(len(CLASS_NAMES))}
    })
# ----------------------------------------------------
# Add tags (user-generated hashtags)
# ----------------------------------------------------
@app.route("/add_tags/<int:id>", methods=["POST"])
@login_required
def add_tags(id):
    tags_raw = request.form.get("tags", "")
    parts = []

    for t in [x.strip() for x in tags_raw.replace(",", " ").split()]:
        if not t:
            continue
        if t.startswith("#"):
            t = t[1:]
        safe = "".join(ch for ch in t if ch.isalnum() or ch in ("-","_"))
        if safe:
            parts.append(safe)

    tags_to_store = ",".join(parts)

    db = get_db()
    row = db.execute(
        "SELECT id FROM predictions WHERE id=? AND user_id=?",
        (id, session["user_id"])
    ).fetchone()

    if not row:
        return jsonify({"error": "Not found or unauthorized"}), 404

    db.execute(
        "UPDATE predictions SET tags=? WHERE id=? AND user_id=?",
        (tags_to_store, id, session["user_id"])
    )
    db.commit()

    return jsonify({"success": True, "tags": tags_to_store})


# ----------------------------------------------------
# CSV EXPORT
# ----------------------------------------------------
@app.route("/export_csv")
@login_required
def export_csv():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY predicted_at DESC",
        (session["user_id"],)
    ).fetchall()

    if not rows:
        return redirect("/history")

    output = io.StringIO()
    writer = csv.writer(output)

    header = rows[0].keys()
    writer.writerow(header)

    for r in rows:
        writer.writerow([r[k] for k in header])

    mem = io.BytesIO(output.getvalue().encode())
    mem.seek(0)

    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="plantguard_history.csv"
    )


# ----------------------------------------------------
# PDF EXPORT
# ----------------------------------------------------
@app.route("/export_pdf/<int:id>")
@login_required
def export_pdf(id):
    if canvas is None:
        return redirect("/history")

    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE id=? AND user_id=?",
        (id, session["user_id"])
    ).fetchone()

    if not row:
        return redirect("/history")

    rec = dict(row)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40

    # HEADER
    c.setFillColor(colors.HexColor("#0b3d2e"))
    c.rect(0, h - 80, w, 80, fill=True)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, h - 50, "PlantGuard AI — Disease Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 68, f"Report ID: {id}")
    c.drawString(180, h - 68, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y = h - 120

    # IMAGE SECTION
    img_path = rec["image_path"]
    iw2 = ih2 = 0
    image_bottom = y

    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            maxw, maxh = 240, 180
            ratio = min(maxw / iw, maxh / ih)
            iw2, ih2 = iw * ratio, ih * ratio

            temp = io.BytesIO()
            img.save(temp, "PNG")
            temp.seek(0)

            from reportlab.lib.utils import ImageReader
            image_reader = ImageReader(temp)

            image_x = 40
            image_y = y - ih2
            c.drawImage(image_reader, image_x, image_y, width=iw2, height=ih2)

            image_bottom = image_y
        except Exception as e:
            print("PDF IMAGE ERROR:", e)

    # META SECTION
    gap = 20
    table_x = (image_x + iw2 + gap) if iw2 > 0 else 40
    table_y = y

    c.setFillColor(colors.black)

    def meta(label, value):
        nonlocal table_y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(table_x, table_y, label + ":")
        c.setFont("Helvetica", 11)
        c.drawString(table_x + 120, table_y, value if value else "—")
        table_y -= 18

    meta("Disease", rec.get("predicted_class"))
    meta("Plant", rec.get("plant_name"))
    meta("Severity", rec.get("severity"))
    meta("Confidence", f"{rec.get('confidence',0)*100:.2f}%")
    meta("Soil", rec.get("soil"))
    meta("Weather", rec.get("weather"))
    meta("Metadata Source", rec.get("metadata_source"))
    meta("Predicted At", str(rec.get("predicted_at")))
    meta("Tags", rec.get("tags"))

    y = min(table_y, image_bottom) - 20

    def section(title, text):
        nonlocal y
        if not text:
            return
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.HexColor("#0b3d2e"))
        c.drawString(40, y, title)
        y -= 16
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 11)
        for line in text.split("\n"):
            c.drawString(55, y, "• " + line.strip())
            y -= 14
        y -= 10

    section("Solution", rec.get("solution"))
    section("Agronomic Tips", rec.get("agronomic") or rec.get("tips"))
    section("Chemicals / Biochemicals", rec.get("chemicals") or rec.get("biochemical"))
    section("Recommended Dosage", rec.get("dosage"))
    section("Preventive Measures", rec.get("preventive"))
    section("Spray Interval", rec.get("spray_interval"))
    section("Organic Alternatives", rec.get("organic"))

    # FOOTER
    c.setFillColor(colors.darkgray)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 30, "Generated by PlantGuard AI — Early Detection System for Perennial Plants")

    c.save()
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"PlantGuard_Report_{id}.pdf"
    )


# ----------------------------------------------------
# QR CODE GENERATION
# ----------------------------------------------------
@app.route("/qr/<int:id>")
@login_required
def qr_code(id):
    url = url_for("view", id=id, _external=True)

    if qrcode is None:
        return redirect(f"/view/{id}")

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ----------------------------------------------------
# MAIN RUN BLOCK
# ----------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        ensure_tables_and_columns()
        fix_db_image_paths()

    app.run(host="0.0.0.0", port=8000, debug=True)
