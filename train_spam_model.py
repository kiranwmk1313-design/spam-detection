import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_ARCHIVE = DATA_DIR / "sms_spam_collection.zip"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_ARCHIVE.is_file():
        archive_bytes = DATASET_ARCHIVE.read_bytes()
    else:
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "spam-malware-student-project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        DATASET_ARCHIVE.write_bytes(archive_bytes)
    return archive_bytes, hashlib.sha256(archive_bytes).hexdigest()


def read_dataset(archive_bytes):
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        dataset_member = next(
            (name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"),
            None,
        )
        if dataset_member is None:
            raise RuntimeError("The UCI archive does not contain SMSSpamCollection.")
        with archive.open(dataset_member) as dataset_file:
            text_stream = io.TextIOWrapper(dataset_file, encoding="utf-8")
            rows = csv.reader(text_stream, delimiter="\t")
            messages = []
            labels = []
            for row in rows:
                if len(row) < 2:
                    continue
                label = row[0].strip().lower()
                if label not in {"ham", "spam"}:
                    continue
                messages.append("\t".join(row[1:]).strip())
                labels.append(int(label == "spam"))
    if len(messages) < 100 or len(set(labels)) != 2:
        raise RuntimeError("The downloaded SMS dataset is missing labeled examples.")
    return messages, np.asarray(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train the SMS spam classifier on the UCI SMS Spam Collection.")
    parser.parse_args()
    archive_bytes, dataset_hash = download_dataset()
    messages, labels = read_dataset(archive_bytes)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    fit_texts, validation_texts, fit_labels, validation_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.2,
        random_state=43,
        stratify=train_labels,
    )

    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                lowercase=True,
                                strip_accents="unicode",
                                ngram_range=(1, 2),
                                max_features=30000,
                                min_df=2,
                                sublinear_tf=True,
                                dtype=np.float32,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=40000,
                                min_df=2,
                                sublinear_tf=True,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=2.0,
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(fit_texts, fit_labels)
    validation_probabilities = model.predict_proba(validation_texts)[:, list(model.classes_).index(1)]
    validation_thresholds = []
    all_thresholds = []
    for threshold in np.linspace(0.05, 0.95, 181):
        validation_predictions = (validation_probabilities >= threshold).astype(np.int32)
        validation_precision = precision_score(validation_labels, validation_predictions, zero_division=0)
        validation_recall = recall_score(validation_labels, validation_predictions, zero_division=0)
        validation_f1 = f1_score(validation_labels, validation_predictions, zero_division=0)
        result = (
            float(validation_f1),
            float(validation_recall),
            float(validation_precision),
            float(threshold),
        )
        all_thresholds.append(result)
        if validation_precision >= 0.9:
            validation_thresholds.append(result)

    selected_threshold = max(validation_thresholds or all_thresholds)[3]
    selected_validation = max(
        (result for result in all_thresholds if result[3] == selected_threshold),
        key=lambda result: result[0],
    )
    model.fit(train_texts, train_labels)
    probabilities = model.predict_proba(test_texts)[:, list(model.classes_).index(1)]
    predicted_labels = (probabilities >= selected_threshold).astype(np.int32)
    matrix = confusion_matrix(test_labels, predicted_labels, labels=[0, 1])
    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "dataset_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "dataset_sha256": dataset_hash,
        "dataset_samples": len(messages),
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "test_class_counts": {
            "ham": int(np.sum(test_labels == 0)),
            "spam": int(np.sum(test_labels == 1)),
        },
        "model": "Word/character TF-IDF + class-balanced LogisticRegression",
        "threshold": selected_threshold,
        "threshold_selection": "Best validation F1 with validation precision >= 0.90; validation-only fallback if unavailable",
        "validation_samples": len(validation_labels),
        "validation_precision": selected_validation[2],
        "validation_recall": selected_validation[1],
        "validation_f1": selected_validation[0],
        "accuracy": float(accuracy_score(test_labels, predicted_labels)),
        "precision": float(precision_score(test_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(test_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(test_labels, predicted_labels, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["Ham", "Spam"],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": selected_threshold}, MODEL_DIR / "spam_model.joblib")
    with (MODEL_DIR / "spam_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="YlGnBu")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        xlabel="Predicted label",
        ylabel="True label",
        title="UCI SMS Spam test set",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(STATIC_DIR / "spam_confusion_matrix.png", dpi=140)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print("Saved spam model, metrics, and confusion matrix.")


if __name__ == "__main__":
    main()