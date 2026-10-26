"""Train/evaluate five classifiers on the public Wisconsin Diagnostic dataset."""
from __future__ import annotations
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "diagnosis"
NAMES = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors",
         "Gaussian Naive Bayes", "Support Vector Machine"]


def slugify(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def build_models():
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline([("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "Gaussian Naive Bayes": Pipeline([("scaler", StandardScaler()),
            ("classifier", GaussianNB())]),
        "Support Vector Machine": Pipeline([("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                               random_state=RANDOM_STATE))]),
    }


def score_positive(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, list(model.classes_).index(1)]
    scores = model.decision_function(X)
    return -scores if list(model.classes_).index(1) == 0 else scores


def evaluate(model, X, y):
    pred = model.predict(X)
    score = score_positive(model, X)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # scikit-learn's target 0 is malignant and 1 is benign.
    y = np.where(dataset.target.to_numpy() == 0, 1, 0).astype(int)
    labels = pd.Series(np.where(y == 1, "malignant", "benign"), name=TARGET)
    X_train, X_test, y_train, y_test, label_train, label_test = train_test_split(
        X, y, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    models = build_models()
    for name in NAMES:
        model = models[name]
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / (slugify(name) + ".joblib"), compress=3)
        row = {"Model": name, **evaluate(model, X_test, y_test)}
        rows.append(row)
        print(name + ": accuracy=" + format(row["Accuracy"], ".4f") +
              ", AUC=" + format(row["AUC"], ".4f"))
    test_frame = X_test.reset_index(drop=True).copy()
    test_frame[TARGET] = label_test.reset_index(drop=True).to_numpy()
    test_frame.to_csv(ROOT / "test_data.csv", index=False, float_format="%.10g")
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "evaluation_results.csv", index=False, float_format="%.6f")
    results.sort_values(["AUC", "F1"], ascending=False).to_csv(
        ROOT / "evaluation_results_ranked.csv", index=False, float_format="%.6f"
    )
    manifest = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "n_instances": int(len(X)), "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_size_fraction": TEST_SIZE, "random_state": RANDOM_STATE,
        "target_column": TARGET, "positive_class": "malignant",
        "class_mapping": {"benign": 0, "malignant": 1},
        "models": {name: "model/" + slugify(name) + ".joblib" for name in NAMES},
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nEvaluation on the fixed stratified holdout set:")
    print(results.to_string(index=False, formatters={c: "{:.4f}".format
          for c in results.columns if c != "Model"}))
    print("Saved " + str(len(test_frame)) + " rows to test_data.csv")


if __name__ == "__main__":
    main()
