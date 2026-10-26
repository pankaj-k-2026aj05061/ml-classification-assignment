"""Check the test CSV and saved classifier files."""
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


class ModelArtifactTests(unittest.TestCase):
    def test_test_csv_has_30_features_and_target(self):
        path = ROOT / "test_data.csv"
        if not path.exists():
            self.skipTest("Run train_models.py first to create test_data.csv.")
        data = pd.read_csv(path)
        self.assertEqual(len(data), 114)
        self.assertIn("diagnosis", data.columns)
        self.assertEqual(len(data.columns) - 1, 30)
        self.assertEqual(set(data["diagnosis"].str.lower()), {"benign", "malignant"})

    def test_saved_models_load_and_predict(self):
        data_path = ROOT / "test_data.csv"
        if not data_path.exists() or not (MODEL_DIR / "manifest.json").exists():
            self.skipTest("Run train_models.py first to save the models.")
        data = pd.read_csv(data_path)
        X = data.drop(columns=["diagnosis"])
        expected = {
            "logistic_regression.joblib", "decision_tree.joblib",
            "k_nearest_neighbors.joblib", "gaussian_naive_bayes.joblib",
            "support_vector_machine.joblib",
        }
        for filename in expected:
            with self.subTest(filename=filename):
                model = joblib.load(MODEL_DIR / filename)
                preds = model.predict(X)
                self.assertEqual(len(preds), len(data))
                self.assertTrue(set(preds).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
