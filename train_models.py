"""Train and compare five classifiers on the IBM Telco Customer Churn sample."""
from __future__ import annotations
import json
import platform
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Gaussian Naive Bayes", "Support Vector Machine"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
DATASET_URL = "https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv"
DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def slugify(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def ensure_dataset():
    if DATA_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the public IBM sample dataset...")
    request = urllib.request.Request(
        DATASET_DOWNLOAD_URL,
        headers={"User-Agent": "ML-Classification-Assignment/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if len(content) < 100_000:
        raise RuntimeError("The dataset download appears incomplete; please retry.")
    DATA_FILE.write_bytes(content)
    print("Dataset downloaded to " + str(DATA_FILE))


def load_dataset():
    ensure_dataset()
    raw = pd.read_csv(DATA_FILE)
    required = {"customerID", TARGET, *NUMERIC_FEATURES}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if len(raw) < 500:
        raise ValueError("The assignment requires at least 500 instances.")
    if raw.shape[1] != 21 or len(raw) != 7043:
        raise ValueError("Unexpected dataset shape; expected the IBM sample (7,043 rows and 21 columns).")
    y = raw[TARGET].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only Yes/No labels.")
    # customerID is an identifier, not a predictor; Churn is the target.
    X = raw.drop(columns=["customerID", TARGET]).copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            raise ValueError("Unexpected missing or non-numeric values in " + col)
    if len(X.columns) < 12:
        raise ValueError("The assignment requires at least 12 predictor features.")
    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]
    for col in categorical_features:
        X[col] = X[col].astype("object")
    return raw, X, y.astype(int), categorical_features


def build_preprocessor(categorical_features, scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, NUMERIC_FEATURES),
         ("categorical", categorical_pipeline, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(categorical_features):
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, True)),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, False)),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, True)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, True)),
            ("classifier", GaussianNB(var_smoothing=1e-9)),
        ]),
        "Support Vector Machine": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, True)),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def positive_class_scores(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def evaluate(model, X, y):
    predicted = model.predict(X)
    scores = positive_class_scores(model, X)
    return {
        "Accuracy": float(accuracy_score(y, predicted)),
        "AUC": float(roc_auc_score(y, scores)),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "F1": float(f1_score(y, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predicted)),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw, X, y, categorical_features = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    models = build_models(categorical_features)
    rows = []
    for name in MODEL_NAMES:
        model = models[name]
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        metrics = evaluate(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        print(name + ": Accuracy=" + format(metrics["Accuracy"], ".4f") + ", AUC=" + format(metrics["AUC"], ".4f"))

    test_data = X_test.reset_index(drop=True).copy()
    test_data[TARGET] = np.where(y_test.reset_index(drop=True).to_numpy() == 1, "Yes", "No")
    test_data.to_csv(ROOT / "test_data.csv", index=False, float_format="%.10g")

    results = pd.DataFrame(rows)
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    results.to_csv(ROOT / "evaluation_results.csv", index=False, float_format="%.6f")
    ranked = results.copy()
    ranked["Average Rank"] = ranked[metric_columns].rank(ascending=False, method="average").mean(axis=1)
    ranked = ranked.sort_values(["Average Rank", "AUC", "F1"], ascending=[True, False, False])
    ranked.to_csv(ROOT / "evaluation_results_ranked.csv", index=False, float_format="%.6f")
    winner = str(ranked.iloc[0]["Model"])

    counts = raw[TARGET].astype(str).str.strip().str.title().value_counts().to_dict()
    manifest = {
        "dataset": "IBM Telco Customer Churn sample",
        "dataset_source": DATASET_URL,
        "dataset_note": "IBM sample data for a fictional California telecommunications company; not actual customer records.",
        "data_file": "data/Telco-Customer-Churn.csv",
        "n_instances": int(len(raw)),
        "n_raw_columns": int(raw.shape[1]),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": categorical_features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target_column": TARGET,
        "positive_class": "Yes (customer churned)",
        "class_mapping": {"No": 0, "Yes": 1},
        "class_balance": {str(k): int(v) for k, v in counts.items()},
        "models": {name: "model/" + MODEL_FILES[name] for name in MODEL_NAMES},
        "evaluation_metrics": metric_columns,
        "overall_winner_by_average_rank": winner,
        "metrics": {str(row["Model"]): {k: float(row[k]) for k in metric_columns} for row in rows},
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nEvaluation on the same stratified test set:")
    print(results.to_string(index=False, formatters={c: "{:.4f}".format for c in metric_columns}))
    print("\nDataset rows: " + str(len(raw)))
    print("Predictor features used: " + str(X.shape[1]))
    print("Training rows: " + str(len(X_train)))
    print("Test rows: " + str(len(X_test)))
    print("Saved models: " + str(len(models)))
    print("Overall winner by average rank across six metrics: " + winner)
    print("Notebook run complete.")


if __name__ == "__main__":
    main()
