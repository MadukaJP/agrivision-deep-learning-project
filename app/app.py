"""
app/app.py — AgriVision AI Flask app (Render deployment)

Local run:  python app.py
Render deployment: set Start Command to `gunicorn app:app` in Render's
dashboard, with this file + requirements.txt + templates/ + static/ in
the repo. The trained model lives once in models/saved_models/ (repo
root's shared copy) — this file references it via a relative path rather
than a duplicated copy inside app/.
"""

import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from fpdf import FPDF
from flask import Flask, render_template, request, send_file, url_for

from treatment_data import get_treatment, UNCERTAIN_MESSAGE

app = Flask(__name__)

# Build paths relative to this file's own location, not the current working
# directory — this way it works the same whether you run `python app.py`
# from inside app/, or Render starts gunicorn from a different working dir.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "saved_models", "agrivision_final.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "..", "models", "saved_models", "class_index.json")
IMG_SIZE = (300, 300)
OOD_CONFIDENCE_THRESHOLD = 0.60

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load once at startup — not per-request, so predictions stay fast.
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}


def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict(img_array):
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    return idx_to_class[top_idx], confidence


def make_gradcam_heatmap(img_array, last_conv_layer_name="top_conv"):
    base = model.get_layer(index=1)
    grad_model = tf.keras.models.Model(
        inputs=base.inputs,
        outputs=[base.get_layer(last_conv_layer_name).output, base.output],
    )
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_array * 255.0)
    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(preprocessed)
        x = base_out
        for layer in model.layers[2:]:
            x = layer(x)
        pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    img = np.array(pil_image.resize(IMG_SIZE)).astype("uint8")
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, IMG_SIZE)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_color * alpha + img * (1 - alpha)).astype("uint8")
    return overlay


def generate_pdf_report(class_name, confidence, treatment, image_path, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AgriVision AI - Diagnostic Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Diagnosis: {class_name.replace('_', ' ').title()}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Confidence: {confidence * 100:.1f}%", ln=True)
    pdf.ln(4)

    if image_path and os.path.exists(image_path):
        pdf.image(image_path, w=100)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Treatment:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, treatment)

    pdf.output(output_path)
    return output_path


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "leaf_image" not in request.files or request.files["leaf_image"].filename == "":
        return render_template("index.html", error="Please choose an image to upload.")

    file = request.files["leaf_image"]
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_path = os.path.join(UPLOAD_DIR, f"{session_id}_original.jpg")

    pil_image = Image.open(file.stream)
    pil_image.convert("RGB").save(original_path)

    img_array, resized_pil = preprocess_image(pil_image)
    class_name, confidence = predict(img_array)

    if confidence < OOD_CONFIDENCE_THRESHOLD:
        return render_template(
            "index.html",
            error=f"Low confidence ({confidence*100:.1f}%). {UNCERTAIN_MESSAGE}",
        )

    heatmap = make_gradcam_heatmap(img_array)
    overlay = overlay_heatmap(resized_pil, heatmap)
    gradcam_filename = f"{session_id}_gradcam.jpg"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)
    Image.fromarray(overlay).save(gradcam_path)

    treatment = get_treatment(class_name)

    pdf_filename = f"{session_id}_report.pdf"
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    generate_pdf_report(class_name, confidence, treatment, original_path, pdf_path)

    return render_template(
        "result.html",
        class_name=class_name.replace("_", " ").title(),
        confidence=f"{confidence * 100:.1f}",
        treatment=treatment,
        original_image=url_for("static", filename=f"uploads/{session_id}_original.jpg"),
        gradcam_image=url_for("static", filename=f"uploads/{gradcam_filename}"),
        pdf_filename=pdf_filename,
    )


@app.route("/download/<filename>")
def download_report(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    return send_file(path, as_attachment=True, download_name="agrivision_report.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
