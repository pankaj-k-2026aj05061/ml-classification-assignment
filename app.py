"""Interactive Streamlit app for the five telco churn classifiers."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "Churn"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABEL_MAP = {"no": 0, "n": 0, "0": 0, "0.0": 0, "false": 0, "yes": 1, "y": 1, "1": 1, "1.0": 1, "true": 1}
st.set_page_config(page_title="Telco Churn Model Evaluator", page_icon="📊", layout="wide")


@st.cache_resource
def load_artifacts():
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Model artifacts are not ready yet. Check that the project training workflow has finished.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {}
    missing = []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(filename)
        else:
            models[name] = joblib.load(path)
    if missing:
        raise FileNotFoundError("Missing saved model files: " + ", ".join(missing))
    return models, manifest


@st.cache_data
def sample_data():
    path = ROOT / "test_data.csv"
    if not path.exists():
        raise FileNotFoundError("test_data.csv is not available yet.")
    return pd.read_csv(path)


def encode_labels(series):
    normalized = series.astype(str).str.strip().str.lower()
    encoded = normalized.map(LABEL_MAP)
    if encoded.isna().any():
        values = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError("Unknown Churn labels: " + ", ".join(values) + ". Use Yes/No or 1/0.")
    return encoded.astype(int).to_numpy()


def prepare_features(data, feature_columns, numeric_features):
    X = data.loc[:, feature_columns].copy()
    for col in numeric_features:
        original = X[col].replace(r"^\s*$", np.nan, regex=True)
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError("Column " + col + " contains non-numeric values.")
        if col == "TotalCharges":
            converted = converted.fillna(0.0)
        elif converted.isna().any():
            raise ValueError("Column " + col + " contains missing numeric values.")
        X[col] = converted
    return X


def positive_class_scores(model, X):
    probability = model.predict_proba(X)
    classes = list(model.classes_)
    return probability[:, classes.index(1)]


def evaluate(model, X, y):
    prediction = model.predict(X)
    scores = positive_class_scores(model, X)
    if len(np.unique(y)) < 2:
        raise ValueError("The Churn column must contain both Yes and No for AUC.")
    metrics = {
        "Accuracy": float(accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, scores)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Recall": float(recall_score(y, prediction, zero_division=0)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, prediction)),
    }
    return metrics, np.asarray(prediction), scores


def main():
    st.title("Telco Customer Churn — Model Evaluator")
    st.caption("Five classifiers • Six required metrics • Upload a labelled CSV to compare predictions")
    st.info("Dataset note: IBM sample data describing a fictional California telecom company. This app is an educational demonstration, not a production retention decision system.")
    try:
        models, manifest = load_artifacts()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    feature_columns = manifest["feature_columns"]
    numeric_features = manifest["numeric_features"]
    with st.sidebar:
        st.header("Dataset")
        st.write("Original records: " + format(manifest["n_instances"], ","))
        st.write("Predictor features: " + str(manifest["n_features"]))
        st.write("Training rows: " + format(manifest["n_train"], ","))
        st.write("Test rows: " + format(manifest["n_test"], ","))
        st.write("Target: Churn (Yes = churned)")
        st.markdown("[IBM sample dataset source](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)")
        st.caption("Upload labelled test data with the 19 predictor columns and a Churn column. CustomerID is not used as a predictor.")

    uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"], max_upload_size=20)
    try:
        data = pd.read_csv(uploaded) if uploaded is not None else sample_data()
    except Exception as exc:
        st.error("Could not read the CSV: " + str(exc))
        st.stop()
    required_columns = feature_columns + [TARGET]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.download_button("Download test CSV template", sample_data().to_csv(index=False).encode("utf-8"), file_name="test_data_template.csv", mime="text/csv")
        st.stop()
    if data.empty:
        st.error("The uploaded CSV contains no data rows.")
        st.stop()
    try:
        X = prepare_features(data, feature_columns, numeric_features)
        y = encode_labels(data[TARGET])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if len(np.unique(y)) < 2:
        st.error("The Churn column must include both Yes and No labels to calculate AUC.")
        st.stop()

    st.success("Ready to evaluate " + format(len(data), ",") + " labelled test rows. The saved models will not be retrained.")
    with st.expander("Preview test data"):
        st.dataframe(data.head(10), use_container_width=True)
        st.write("Stayed (No): " + format(int((y == 0).sum()), ",") + " | Churned (Yes): " + format(int((y == 1).sum()), ","))

    comparison_rows = []
    details = {}
    for name, model in models.items():
        metrics, prediction, scores = evaluate(model, X, y)
        comparison_rows.append({"Model": name, **metrics})
        details[name] = (metrics, prediction, scores)
    comparison = pd.DataFrame(comparison_rows).sort_values(["AUC", "F1"], ascending=False).reset_index(drop=True)
    st.subheader("Model comparison")
    st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "Model"}), use_container_width=True, hide_index=True)
    top_auc = comparison.iloc[0]
    st.write("Highest AUC on this uploaded test file: " + str(top_auc["Model"]) + " (AUC " + format(top_auc["AUC"], ".4f") + ").")

    st.subheader("Inspect one model")
    selected_name = st.selectbox("Choose classifier", list(models.keys()))
    metrics, prediction, scores = details[selected_name]
    columns = st.columns(6)
    for column, metric in zip(columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        column.metric(metric, format(metrics[metric], ".4f"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        matrix = confusion_matrix(y, prediction, labels=[0, 1])
        st.dataframe(pd.DataFrame(matrix, index=["Actual stayed (No)", "Actual churned (Yes)"], columns=["Predicted stayed (No)", "Predicted churned (Yes)"]), use_container_width=True)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, prediction, labels=[0, 1], target_names=["Stayed (No)", "Churned (Yes)"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    output = pd.DataFrame({
        "actual_churn": np.where(y == 1, "Yes", "No"),
        "predicted_churn": np.where(prediction == 1, "Yes", "No"),
        "churn_probability": scores,
    })
    st.download_button("Download selected-model predictions", output.to_csv(index=False).encode("utf-8"), file_name="predictions_" + selected_name.lower().replace(" ", "_") + ".csv", mime="text/csv")


if __name__ == "__main__":
    main()
