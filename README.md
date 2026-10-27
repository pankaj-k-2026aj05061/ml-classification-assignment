# Telco Customer Churn — Five-Model Classification Assignment

**GitHub repository:** https://github.com/pankaj-k-2026aj05061/ml-classification-assignment  
**Live Streamlit app:** https://pankaj-k-2026aj05061-ml-classification-assignment-app-t6l9gk.streamlit.app/

## A. Problem statement

Customer churn occurs when a customer stops using a service. This project compares five standard machine-learning classifiers on a public telecom churn sample to understand how accurately the models identify customers recorded as having left. It is an educational comparison, not a production retention system or a basis for decisions about actual customers.

All five classifiers use the same stratified 80/20 train/test split. Accuracy, AUC, precision, recall, F1, and Matthews correlation coefficient (MCC) are calculated on the same test rows.

## B. Dataset description

- **Dataset:** IBM Telco Customer Churn sample.
- **Source:** [IBM's public sample dataset on GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).
- **Size:** 7,043 customer records and 21 original columns.
- **Important source note:** IBM describes this as sample data for a fictional California telecommunications company. It illustrates a realistic churn-prediction use case, but it is not actual customer data from a real telecom provider.
- **Target:** Churn, with No encoded as 0 and Yes encoded as 1 (the positive class).
- **Predictors:** 19 features after removing customerID (an identifier) and the Churn target.
- **Target balance:** 5,174 No records and 1,869 Yes records in the full dataset.
- **Preprocessing:** TotalCharges is converted to numeric; its 11 blank values are set to 0 because those records have zero tenure. Numeric features are imputed if needed and standardized for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. The Decision Tree uses numeric features without scaling. Categorical features are imputed and one-hot encoded within each model pipeline. All preprocessing is fitted on training rows only.
- **Split:** Stratified 80/20 split with random_state=42. The script writes the exact training/test row counts to model/manifest.json. The expected split is 5,634 training rows and 1,409 test rows.

The script downloads the dataset into the local data folder on first run if it is not already present. The app evaluates test_data.csv, which contains the 19 predictor columns and a labelled Churn target.

## C. GitHub repository link

https://github.com/pankaj-k-2026aj05061/ml-classification-assignment

## Live Streamlit app link

https://pankaj-k-2026aj05061-ml-classification-assignment-app-t6l9gk.streamlit.app/

## D. Models and evaluation metrics

All five models are evaluated on the same stratified test set. AUC uses the estimated probability of the positive class (Churn = Yes). Precision, recall, and F1 treat a churned customer (Yes) as the positive class.

The exact metric values and observations will be inserted from the first successful, reproducible training run before the submission is prepared. The definitive results are also saved in evaluation_results.csv.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Pending initial run | — | — | — | — | — |
| Decision Tree | Pending initial run | — | — | — | — | — |
| K-Nearest Neighbors | Pending initial run | — | — | — | — | — |
| Gaussian Naive Bayes | Pending initial run | — | — | — | — | — |
| Support Vector Machine | Pending initial run | — | — | — | — | — |

### Observations about model performance

The final README will describe each model using the measured test-set results, including false-positive and false-negative trade-offs where relevant. The overall winner will be selected by average rank across all six metrics, with AUC and F1 used only to break ties. This avoids selecting a winner based on accuracy alone.

| Model | Observation about performance |
|---|---|
| Logistic Regression | To be completed from the measured metrics after training. |
| Decision Tree | To be completed from the measured metrics after training. |
| K-Nearest Neighbors | To be completed from the measured metrics after training. |
| Gaussian Naive Bayes | To be completed from the measured metrics after training. |
| Support Vector Machine | To be completed from the measured metrics after training. |
| Overall winner | To be completed from the average rank across Accuracy, AUC, Precision, Recall, F1, and MCC. |

## Streamlit app features

- Load the bundled labelled test dataset or upload a CSV.
- Compare all five saved classifiers using the six required metrics.
- Inspect a selected model's confusion matrix and classification report.
- Download predicted churn labels and churn probabilities.
- Validate required feature columns and accepted target labels (Yes/No or 1/0).

The uploaded CSV must contain the same 19 predictor columns as test_data.csv plus a Churn column. Do not include customerID as a required predictor. Both Churn classes must be present to calculate AUC.

## Repository files

- app.py — interactive Streamlit application.
- train_models.py — dataset retrieval, cleaning, stratified split, training, metric calculation, and artifact creation.
- requirements.txt — required Python dependencies.
- test_data.csv — labelled test rows generated by the training script.
- model/ — five saved model pipelines and manifest.json.
- evaluation_results.csv and evaluation_results_ranked.csv — metric comparison and average-rank table.
- tests/test_models.py — regression tests for the saved models and test data.
- .github/workflows/train-and-save-models.yml — trains, tests, and saves model outputs.
- BITS_Virtual_Lab_Execution.ipynb — notebook for the required BITS Virtual Lab run.

## Run locally or in BITS Virtual Lab

1. Open a terminal in the project directory.
2. Install dependencies with python -m pip install -r requirements.txt.
3. Run python train_models.py to download the public dataset if needed, fit all five models, and generate the labelled test CSV and metric tables.
4. Run python -m unittest discover -s tests -v to check saved model artifacts.
5. Run streamlit run app.py to launch the interactive app.

In BITS Virtual Lab, open BITS_Virtual_Lab_Execution.ipynb from the project directory and run all cells. It executes the same training script and displays the generated metrics and output-file counts. Capture the requested screenshot inside the BITS Virtual Lab environment.
