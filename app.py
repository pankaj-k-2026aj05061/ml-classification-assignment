"""Compare the saved classifiers on a labeled test CSV."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABEL_MAP = {"benign": 0, "b": 0, "0": 0, "0.0": 0,
             "malignant": 1, "m": 1, "1": 1, "1.0": 1}

st.set_page_config(page_title="ML Classification Model Evaluator", page_icon="📊", layout="wide")


@st.cache_resource
def load_models():
    loaded = {}
    missing = []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(filename)
        else:
            loaded[name] = joblib.load(path)
    if missing:
        raise FileNotFoundError("Saved model files are not ready yet: " + ", ".join(missing)
                                + ". Check the latest run under GitHub Actions.")
    return loaded


@st.cache_data
def sample_data():
    path = ROOT / "test_data.csv"
    if not path.exists():
        raise FileNotFoundError("test_data.csv was not found. Run train_models.py to create it.")
    return pd.read_csv(path)


def encode_labels(series):
    normalized = series.astype(str).str.strip().str.lower()
    encoded = normalized.map(LABEL_MAP)
    if encoded.isna().any():
        values = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError("Unrecognized diagnosis labels: " + ", ".join(values))
    return encoded.astype(int).to_numpy()


def malignant_scores(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        return probs[:, classes.index(1)]
    scores = np.asarray(model.decision_function(X))
    if list(model.classes_).index(1) == 0:
        scores = -scores
    return scores


def evaluate(model, X, y):
    pred = model.predict(X)
    score = malignant_scores(model, X)
    metrics = {
        "Accuracy": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }
    return metrics, np.asarray(pred), score


def main():
    st.title("ML Classification Model Evaluator")
    st.caption("Five saved classifiers • Six evaluation metrics • Labeled CSV test data")
    st.warning("Educational ML demonstration only. Not intended for medical or clinical decisions.")
    try:
        models = load_models()
        manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
        feature_names = list(next(iter(models.values())).feature_names_in_)
    except Exception as exc:
        st.error(str(exc))
        st.info("The model files are created by the GitHub Actions workflow. Check the latest run, then refresh this app.")
        st.stop()

    with st.sidebar:
        st.header("Dataset")
        st.write("UCI Breast Cancer Wisconsin (Diagnostic)")
        st.write("Training rows: " + str(manifest["n_train"]))
        st.write("Holdout rows: " + str(manifest["n_test"]))
        st.write("Features required: " + str(len(feature_names)))
        st.markdown("[UCI dataset source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")
        st.caption("Upload a CSV with all 30 feature columns and a diagnosis column. The app evaluates the file without retraining.")

    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"], max_upload_size=20)
    try:
        data = pd.read_csv(uploaded) if uploaded is not None else sample_data()
    except Exception as exc:
        st.error("Could not read the CSV: " + str(exc))
        st.stop()
    required = feature_names + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.download_button("Download sample CSV template", sample_data().to_csv(index=False),
                           file_name="test_data_template.csv", mime="text/csv")
        st.stop()
    if data.empty:
        st.error("The CSV contains no data rows.")
        st.stop()
    X = data.loc[:, feature_names].copy()
    for col in feature_names:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error("All 30 feature columns must contain finite numeric values with no missing cells.")
        st.stop()
    try:
        y = encode_labels(data[TARGET])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if len(np.unique(y)) < 2:
        st.error("The diagnosis column must include both classes to calculate AUC.")
        st.stop()

    st.success("Ready to evaluate " + str(len(data)) + " labeled rows. No model is retrained.")
    with st.expander("Preview test data"):
        st.dataframe(data.head(10), use_container_width=True)
        st.write("Benign: " + str(int((y == 0).sum())) + " | Malignant: " + str(int((y == 1).sum())))

    all_rows, details = [], {}
    for name, model in models.items():
        metrics, pred, scores = evaluate(model, X, y)
        all_rows.append({"Model": name, **metrics})
        details[name] = (metrics, pred, scores)
    comparison = pd.DataFrame(all_rows).sort_values(["AUC", "F1"], ascending=False).reset_index(drop=True)
    st.subheader("Model comparison")
    st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "Model"}),
                 use_container_width=True, hide_index=True)
    winner = comparison.iloc[0]
    st.write("Highest AUC on this test file: " + str(winner["Model"]) + " (AUC " + format(winner["AUC"], ".4f") + ").")

    st.subheader("Inspect a model")
    selected_name = st.selectbox("Choose classifier", list(models.keys()))
    metrics, pred, scores = details[selected_name]
    kpis = st.columns(6)
    for col, key in zip(kpis, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(key, format(metrics[key], ".4f"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (rows=actual, columns=predicted)")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        st.dataframe(pd.DataFrame(cm, index=["Actual benign", "Actual malignant"],
                                  columns=["Predicted benign", "Predicted malignant"]),
                     use_container_width=True)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, pred, labels=[0, 1],
                                       target_names=["benign", "malignant"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    pred_frame = pd.DataFrame({"actual_diagnosis": np.where(y == 1, "malignant", "benign"),
                               "predicted_diagnosis": np.where(pred == 1, "malignant", "benign"),
                               "malignant_score": scores})
    st.download_button("Download selected-model predictions",
                       pred_frame.to_csv(index=False).encode("utf-8"),
                       file_name="predictions_" + selected_name.lower().replace(" ", "_") + ".csv",
                       mime="text/csv")


if __name__ == "__main__":
    main()
