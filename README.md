# Comparing Five Classification Models

**GitHub repository:** https://github.com/pankaj-k-2026aj05061/ml-classification-assignment  
**Live Streamlit app:** https://pankaj-k-2026aj05061-ml-classification-assignment-app-t6l9gk.streamlit.app/

## A. Problem statement

This project compares five classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset. All models use the same stratified train/test split and are evaluated using accuracy, AUC, precision, recall, F1, and Matthews correlation coefficient (MCC). The Streamlit app evaluates a labeled CSV using the saved models and displays the comparison, confusion matrix, classification report, and downloadable predictions.

## B. Dataset description

The original dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). For reproducibility, the training script loads scikit-learn's bundled copy with **load_breast_cancer(as_frame=True)**. This is why the feature names in the preview use names such as **mean radius**, **mean perimeter**, and **mean area**.

- **Instances and features:** 569 rows and 30 numeric predictor features.
- **Class balance in the full dataset:** 357 benign and 212 malignant cases.
- **Target:** diagnosis, with benign encoded as 0 and malignant encoded as 1 (the positive class).
- **Split:** stratified 80/20 split with **random_state=42**; 455 training rows and 114 holdout rows.
- **Holdout composition:** 72 benign and 42 malignant rows.
- **Preprocessing:** StandardScaler is used in Logistic Regression, KNN, Gaussian Naive Bayes, and SVM pipelines. Decision Tree is not scaled. Preprocessing is fitted on training rows only.

## C. GitHub repository link

https://github.com/pankaj-k-2026aj05061/ml-classification-assignment

## Live Streamlit app link

https://pankaj-k-2026aj05061-ml-classification-assignment-app-t6l9gk.streamlit.app/

## D. Models and evaluation metrics

All values below are from the same fixed 114-row holdout set. AUC uses the score for the positive class (malignant); the other metrics use each model's predicted class. The confusion-matrix counts in the observations use the order **TN, FP, FN, TP**.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Hyperparameters used

| Model | Main settings |
|---|---|
| Logistic Regression | C=1.0, solver=lbfgs, max_iter=5000; standardized features |
| Decision Tree | criterion=gini, max_depth=None (unpruned), min_samples_split=2, min_samples_leaf=1; no scaling |
| K-Nearest Neighbors | n_neighbors=5, weights=uniform, Euclidean distance (p=2); standardized features |
| Gaussian Naive Bayes | var_smoothing=1e-9; standardized features |
| Support Vector Machine | RBF kernel, C=1.0, gamma=scale, probability=True; standardized features |

### Observations about model performance

| Model | Observation |
|---|---|
| Logistic Regression | Correctly classified 110 of 114 rows: TN=71, FP=1, FN=3, TP=39. It has the highest AUC (0.9960) and recall of 0.9286, matching SVM's recall on this split. |
| Decision Tree | Correctly classified 106 of 114 rows: TN=68, FP=4, FN=4, TP=38. Its AUC (0.9246) is the lowest of the five, but its accuracy is not the lowest; Gaussian Naive Bayes has lower accuracy (0.9211). The tree is unpruned (max_depth=None), so it can fit complex patterns in the training data. |
| K-Nearest Neighbors | Correctly classified 109 of 114 rows: TN=71, FP=1, FN=4, TP=38. Precision is strong (0.9744), while recall is 0.9048, meaning it misses four malignant cases in the holdout set. Feature scaling is important for a distance-based model. |
| Gaussian Naive Bayes | Correctly classified 105 of 114 rows: TN=69, FP=3, FN=6, TP=36. It has the lowest accuracy, recall, F1, and MCC in this comparison, and misses six malignant cases. The conditional-independence assumption may be a poor fit for this dataset: mean radius, mean perimeter, and mean area are strongly correlated (pairwise correlations are approximately 0.998, 0.987, and 0.987). |
| Support Vector Machine | Correctly classified 111 of 114 rows: TN=72, FP=0, FN=3, TP=39. It has the highest accuracy (0.9737), precision (1.0000), F1 (0.9630), and MCC (0.9442), with no false positives but three missed malignant cases. It gets only one more test row correct than Logistic Regression. |

**Overall winner for this holdout split:** Support Vector Machine is the best overall choice if the emphasis is on accuracy, precision, F1, MCC, and avoiding false positives. However, it beats Logistic Regression by just one correctly classified row (111 versus 110), and both have recall of 0.9286. Logistic Regression has the highest AUC (0.9960 versus 0.9947 for SVM), so the preferred model depends on the evaluation goal and the relative cost of false positives and false negatives.

**SVM score note:** The SVM predicted label comes from the classifier's decision function, while its exported malignant_score and AUC use the calibrated probability from predict_proba. Scikit-learn's SVC with probability enabled can occasionally return a calibrated probability above 0.5 for a row whose predicted label is benign. This is a known difference between the calibrated probability and the class-prediction rule, not a mismatch in the test row.

## Streamlit app features

- Upload a labeled test CSV or use the bundled test_data.csv.
- Compare all five saved model pipelines using the six required metrics.
- Inspect selected-model metrics, confusion matrix, and classification report.
- Download model predictions and malignant-class scores.
- Input checks for missing features, invalid numeric values, unknown labels, empty files, and single-class target data.

The uploaded CSV must contain the exact 30 feature columns and a diagnosis column. Accepted labels are benign/malignant, B/M, or 0/1 where 0 means benign and 1 means malignant. Both target classes must be present for AUC calculation.

## Repository structure

- app.py: interactive Streamlit application
- train_models.py: training and evaluation script
- requirements.txt: dependencies
- test_data.csv: labeled holdout data
- model/: five saved pipelines and manifest.json
- evaluation_results.csv and evaluation_results_ranked.csv: evaluation tables
- tests/test_models.py: automated checks
- .github/workflows/train-and-save-models.yml: runs syntax checks, training, tests, and a Streamlit startup check
- BITS_Virtual_Lab_Execution.ipynb: notebook used to run the same training script in BITS Virtual Lab

## Run locally

From the project directory, install dependencies with **python -m pip install -r requirements.txt**, train and recreate the results with **python train_models.py**, run the checks with **python -m unittest discover -s tests -v**, and start the app with **streamlit run app.py**.

## BITS Virtual Lab execution

During the BITS Virtual Lab run, BITS_Virtual_Lab_Execution.ipynb executed train_models.py, which fitted all five classifiers on the shared training split, generated the holdout test CSV and metric tables, and printed the results shown in the lab screenshot. GitHub Actions also runs the same script to check reproducibility and keep the saved model artifacts and CSV outputs current.
