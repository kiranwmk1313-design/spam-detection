import json
import hmac
import os
import secrets
from pathlib import Path

import joblib
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from utils.malware_utils import extract_features, is_pe_file


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
AUTH_CONFIG_PATH = BASE_DIR / "instance" / "auth.json"
MAX_TEXT_LENGTH = 5000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".exe", ".dll", ".pdf", ".docx", ".zip", ".txt", ".js"}
MALWARE_MODEL_PATH = MODEL_DIR / "malware_model.joblib"
MALWARE_SCALER_PATH = MODEL_DIR / "malware_scaler.joblib"


def load_auth_configuration():
    environment_username = os.environ.get("SPAM_APP_USERNAME", "")
    environment_password_hash = os.environ.get("SPAM_APP_PASSWORD_HASH", "")
    environment_secret_key = os.environ.get("FLASK_SECRET_KEY", "")
    if environment_username and environment_password_hash and environment_secret_key:
        return environment_username, environment_password_hash, environment_secret_key, "environment"

    try:
        with AUTH_CONFIG_PATH.open(encoding="utf-8") as auth_file:
            settings = json.load(auth_file)
        username = settings.get("username", "")
        password_hash = settings.get("password_hash", "")
        secret_key = settings.get("secret_key", "")
        if username and password_hash and secret_key:
            return username, password_hash, secret_key, "local"
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return "", "", environment_secret_key or secrets.token_urlsafe(48), "unconfigured"


LOGIN_USERNAME, LOGIN_PASSWORD_HASH, SESSION_SECRET_KEY, AUTH_SOURCE = load_auth_configuration()
AUTH_ENABLED = bool(LOGIN_USERNAME and LOGIN_PASSWORD_HASH)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=SESSION_SECRET_KEY,
    MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
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


def save_auth_configuration(username, password):
    AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = AUTH_CONFIG_PATH.with_suffix(".tmp")
    settings = {
        "username": username,
        "password_hash": generate_password_hash(password),
        "secret_key": secrets.token_urlsafe(48),
    }
    with temporary_path.open("w", encoding="utf-8") as auth_file:
        json.dump(settings, auth_file)
    temporary_path.replace(AUTH_CONFIG_PATH)
    return settings


@app.context_processor
def inject_auth_settings():
    return {
        "auth_enabled": AUTH_ENABLED,
        "auth_source": AUTH_SOURCE,
        "csrf_token": session.setdefault("_csrf_token", secrets.token_urlsafe(32)),
    }


def valid_csrf_token():
    expected_token = session.get("_csrf_token", "")
    provided_token = request.form.get("csrf_token", "")
    return bool(expected_token) and hmac.compare_digest(provided_token, expected_token)


@app.before_request
def require_login():
    if not AUTH_ENABLED or request.endpoint in {"login", "static"}:
        return None
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    return None


@app.get("/configure-login")
@app.post("/configure-login")
def configure_login():
    global AUTH_ENABLED, AUTH_SOURCE, LOGIN_PASSWORD_HASH, LOGIN_USERNAME

    if AUTH_SOURCE == "environment":
        flash("Sign-in is managed by environment variables and cannot be changed here.", "notice")
        return redirect(url_for("login"))

    if request.method == "POST":
        if not valid_csrf_token():
            flash("This setup form expired. Please try again.", "error")
            return redirect(url_for("configure_login"))
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password_confirmation = request.form.get("password_confirmation", "")
        if len(username) < 3 or len(username) > 64:
            flash("Username must be between 3 and 64 characters.", "error")
        elif len(password) < 12:
            flash("Choose a password with at least 12 characters.", "error")
        elif password != password_confirmation:
            flash("The password confirmation does not match.", "error")
        else:
            try:
                settings = save_auth_configuration(username, password)
            except OSError:
                flash("Could not save sign-in settings on this device.", "error")
            else:
                LOGIN_USERNAME = settings["username"]
                LOGIN_PASSWORD_HASH = settings["password_hash"]
                app.config["SECRET_KEY"] = settings["secret_key"]
                AUTH_ENABLED = True
                AUTH_SOURCE = "local"
                session.clear()
                flash("Sign-in is configured. Use your new credentials to continue.", "notice")
                return redirect(url_for("login"))
    return render_template("configure_login.html", auth_already_enabled=AUTH_ENABLED)


@app.get("/login")
@app.post("/login")
def login():
    if request.method == "POST":
        if not valid_csrf_token():
            flash("This sign-in form expired. Please try again.", "error")
            return redirect(url_for("login"))
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not AUTH_ENABLED:
            flash("Sign-in is not configured. Use Configure sign-in to create a local account.", "notice")
        else:
            try:
                valid_password = check_password_hash(LOGIN_PASSWORD_HASH, password)
            except (ValueError, TypeError):
                valid_password = False
            if hmac.compare_digest(username, LOGIN_USERNAME) and valid_password:
                session.clear()
                session["authenticated"] = True
                session["username"] = LOGIN_USERNAME
                return redirect(url_for("home"))
            flash("The username or password is incorrect.", "error")
    return render_template("login.html")


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


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