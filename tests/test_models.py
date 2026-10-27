"""Regression checks for saved churn models and held-out test CSV."""
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "logistic_regression.joblib", "decision_tree.joblib", "k_nearest_neighbors.joblib",
    "gaussian_naive_bayes.joblib", "support_vector_machine.joblib",
}


class ChurnModelTests(unittest.TestCase):
    def test_test_csv_has_required_shape_and_classes(self):
        data = pd.read_csv(ROOT / "test_data.csv")
        self.assertEqual(len(data), 1409)
        self.assertIn("Churn", data.columns)
        self.assertEqual(len(data.columns) - 1, 19)
        self.assertEqual(set(data["Churn"].astype(str).str.strip()), {"Yes", "No"})
        self.assertFalse(data.isna().any().any())

    def test_all_five_saved_models_load_and_predict(self):
        manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
        data = pd.read_csv(ROOT / "test_data.csv")
        X = data.loc[:, manifest["feature_columns"]]
        y = data["Churn"].map({"No": 0, "Yes": 1}).to_numpy()
        self.assertEqual(set(MODEL_FILES), {Path(p).name for p in manifest["models"].values()})
        for filename in MODEL_FILES:
            with self.subTest(filename=filename):
                model = joblib.load(MODEL_DIR / filename)
                prediction = model.predict(X)
                probability = model.predict_proba(X)
                self.assertEqual(len(prediction), len(data))
                self.assertTrue(set(np.unique(prediction)).issubset({0, 1}))
                self.assertEqual(probability.shape, (len(data), 2))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(set(np.unique(y)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
