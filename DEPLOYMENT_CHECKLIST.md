# Deployment and submission checklist

1. Confirm the GitHub Actions workflow has succeeded and model/*.joblib plus test_data.csv exist in main.
2. On Streamlit Community Cloud, create an app from this repository, branch main, entry file app.py.
3. Wait for deployment, open the public URL, and test the bundled CSV plus an uploaded CSV.
4. Verify the comparison table, selected-model metrics, confusion matrix, classification report, and prediction download.
5. Open BITS Virtual Lab with your account, run the notebook or training script, and capture a genuine execution screenshot there.
6. Update README with the actual working Streamlit URL.
7. Produce one final PDF containing repository link, live app link, the complete README, and the BITS Virtual Lab screenshot. Do not submit the preview PDF.
