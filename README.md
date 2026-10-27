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
- **Split:** Stratified 80/20 split with random_state=42: 5,634 training rows and 1,409 test rows. The test split contains 1,035 No cases and 374 Yes cases.

The training script downloads the dataset into the local data folder on first run if it is not already present. The app evaluates test_data.csv, which contains the 19 predictor columns and a labelled Churn target.

## C. GitHub repository link

https://github.com/pankaj-k-2026aj05061/ml-classification-assignment

## Live Streamlit app link

https://pankaj-k-2026aj05061-ml-classification-assignment-app-t6l9gk.streamlit.app/

## D. Models and evaluation metrics

All values below are measured on the same 1,409-row stratified test set. AUC uses the estimated probability of the positive class (Churn = Yes). Precision, recall, and F1 treat a churned customer (Yes) as the positive class.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.8421 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7197 | 0.6467 | 0.4730 | 0.4920 | 0.4823 | 0.2903 |
| K-Nearest Neighbors | 0.7637 | 0.7884 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Gaussian Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7913 | 0.7902 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

### Observations about model performance

The confusion matrix counts below use the order TN (correctly predicted No), FP (No predicted as Yes), FN (Yes predicted as No), and TP (correctly predicted Yes).

| Model | Observation about performance |
|---|---|
| Logistic Regression | Best overall on the average-rank rule across the six metrics. It has the highest accuracy (0.8055), AUC (0.8421), precision (0.6572), F1 (0.6040), and MCC (0.4790). On the test set it correctly classifies 1,135 of 1,409 rows (TN=926, FP=109, FN=165, TP=209). Its recall is 0.5588, so it misses 165 of the 374 churn cases. |
| Decision Tree | Lowest AUC (0.6467) and MCC (0.2903), with 1,014 correct predictions (TN=830, FP=205, FN=190, TP=184). Its precision, recall, and F1 are also the lowest or among the lowest, so it is the weakest model on this particular split. |
| K-Nearest Neighbors | Correctly classifies 1,076 rows (TN=861, FP=174, FN=159, TP=215). Recall (0.5749) is slightly higher than Logistic Regression, but precision (0.5527), AUC (0.7884), F1 (0.5636), and MCC (0.4018) are lower. |
| Gaussian Naive Bayes | Highest recall (0.8369): it finds 313 of the 374 churn cases and misses only 61. However, it produces 369 false positives (TN=666, FP=369, FN=61, TP=313), giving the lowest accuracy (0.6948) and precision (0.4589). It may suit a use case where missing a churner is especially costly and a large number of follow-up alerts is acceptable. |
| Support Vector Machine | Correctly classifies 1,115 rows (TN=934, FP=101, FN=193, TP=181). It has the fewest false positives (101) and precision of 0.6418, but recall is only 0.4840, so it misses 193 of the 374 churn cases. Its AUC is 0.7902 and MCC is 0.4264. |
| **Overall winner: Logistic Regression** | It has the best average rank across Accuracy, AUC, Precision, Recall, F1, and MCC. This choice balances the metrics instead of choosing a model based only on accuracy. Gaussian Naive Bayes has the strongest recall, so it could be preferable if detecting as many churn cases as possible matters more than limiting false positives. |

The average-rank comparison is saved in evaluation_results_ranked.csv. Each metric is ranked with higher values considered better; the model with the lowest mean rank is selected. AUC and F1 break ties.

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
