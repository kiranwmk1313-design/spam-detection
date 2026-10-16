# Spam and Malware Detection

Local Flask project for an experimental SMS spam classifier and static malware screening demo.

## Setup on Windows

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the malware model

```powershell
python train_malware_model.py
```

The script downloads EMBER 2018 v2 feature records into `datasets/ember2018/` and verifies each shard against its published SHA-256. It excludes unclassified rows and creates a reproducible, stratified 80/20 holdout from the labeled EMBER training records. By default, it scans up to three consecutive training shards starting at shard 50 and uses at most 1,000 benign and 1,000 malware examples. It stops as soon as both class limits are reached. Adjust `--start-shard`, `--shard-count`, or `--max-per-class` to change the subset; shard downloads with failed checksums are rejected.

The classifier is a feed-forward neural network with 64- and 32-unit ReLU layers, implemented with scikit-learn's `MLPClassifier`. TensorFlow is specified in the original project documents, but Windows Application Control blocked its native DLL in this environment, so the runnable version uses scikit-learn instead.

The dataset contains extracted feature vectors, not executable files. Training uses only the first 256 normalized values, EMBER's raw byte histogram. Upload-time extraction computes the same histogram directly from the uploaded bytes. The model, scaler, metrics JSON, and confusion matrix are saved under `models/` and `static/`.

Dataset source: [EMBER 2018 v2 mirror](https://huggingface.co/datasets/cw1521/ember2018-malware-v2), pinned to revision `8472b3de1c2029bf9376465e834a679d50933136`. Original project: [Elastic/EMBER](https://github.com/elastic/ember). Labels `-1` (benign) and `1` (malware) are used; unclassified rows are excluded.

## Run the app

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Restart the app after training so it loads the saved model. The upload route accepts `.exe` and `.dll` PE files, enforces a 10 MB request limit, validates them with `pefile`, and analyzes bytes in memory without saving or executing the file.

## Sign-in

On a fresh installation, open <http://127.0.0.1:5000/configure-login> and create a username and password. Passwords must contain at least 12 characters. The app stores only a Werkzeug password hash and a generated session key in `instance/auth.json`; `instance/` is excluded from Git. After setup, the app routes require sign-in. Use **Configure sign-in** while logged in to update the account, and **Sign out** to end the session.

For deployments managed outside the app, `SPAM_APP_USERNAME`, `SPAM_APP_PASSWORD_HASH`, and `FLASK_SECRET_KEY` environment variables can configure sign-in instead. Do not use the local development server or expose account configuration to an untrusted network.

The malware score is not a calibrated probability. A byte histogram is a limited signal and may miss novel or modified malware. This experimental project must not replace antivirus or professional security analysis. The spam page remains a scaffold until its dataset and training pipeline are implemented.