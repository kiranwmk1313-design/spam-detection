import json
import os
from pathlib import Path

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for

from utils.malware_utils import extract_features, is_pe_file


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MAX_TEXT_LENGTH = 5000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".exe", ".dll", ".pdf", ".docx", ".zip", ".txt", ".js"}
MALWARE_MODEL_PATH = MODEL_DIR / "malware_model.joblib"
MALWARE_SCALER_PATH = MODEL_DIR / "malware_scaler.joblib"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-development-only-change-me"),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE,
)


def spam_artifacts_exist():
    return (MODEL_DIR / "spam_model.keras").is_file() and (MODEL_DIR / "tokenizer.joblib").is_file()


def load_malware_artifacts():
    if not MALWARE_MODEL_PATH.is_file() or not MALWARE_SCALER_PATH.is_file():
        return None, None
    return joblib.load(MALWARE_MODEL_PATH), joblib.load(MALWARE_SCALER_PATH)


def load_metrics(filename):
    metrics_path = MODEL_DIR / filename
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (OSError, json.JSONDecodeError):
        return None


MALWARE_MODEL, MALWARE_SCALER = load_malware_artifacts()


@app.get("/")
def home():
    return render_template("index.html", malware_model_ready=MALWARE_MODEL is not None)


@app.route("/spam", methods=["GET", "POST"])
def spam():
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "")
        if not message.strip():
            flash("Enter a message to analyze.", "error")
        elif len(message) > MAX_TEXT_LENGTH:
            flash(f"Messages must be {MAX_TEXT_LENGTH:,} characters or fewer.", "error")
        elif not spam_artifacts_exist():
            flash("The spam model is not trained yet. Run train_spam_model.py after adding the dataset.", "notice")
        else:
            flash("Spam prediction is not available yet.", "notice")
    return render_template("spam.html", message=message, max_text_length=MAX_TEXT_LENGTH)


@app.route("/malware", methods=["GET", "POST"])
def malware():
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            flash("Choose a file to analyze.", "error")
        elif Path(uploaded_file.filename).suffix.lower() not in {".exe", ".dll"}:
            flash("Upload a Windows PE executable (.exe or .dll).", "error")
        elif MALWARE_MODEL is None or MALWARE_SCALER is None:
            flash("The malware model is not trained yet. Run train_malware_model.py.", "notice")
        else:
            file_bytes = uploaded_file.stream.read(MAX_UPLOAD_SIZE + 1)
            if len(file_bytes) > MAX_UPLOAD_SIZE:
                flash("The upload exceeds the 10 MB limit.", "error")
            elif not is_pe_file(file_bytes):
                flash("That file is not a valid Windows PE executable. It was not saved or executed.", "error")
            else:
                features, metadata = extract_features(file_bytes)
                scaled_features = MALWARE_SCALER.transform(features)
                malware_score = float(MALWARE_MODEL.predict_proba(scaled_features)[0, 1])
                result = {
                    "label": "Potentially Malicious" if malware_score >= 0.5 else "Benign",
                    "malware_score": malware_score,
                    "metadata": metadata,
                }
    return render_template("malware.html", allowed_extensions=[".exe", ".dll"], result=result)


@app.get("/evaluation")
def evaluation():
    return render_template(
        "evaluation.html",
        spam_metrics=load_metrics("spam_metrics.json"),
        malware_metrics=load_metrics("malware_metrics.json"),
    )


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The upload exceeds the 10 MB limit.", "error")
    return redirect(url_for("malware"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)