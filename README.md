Customer Churn Prediction
A machine learning project that predicts whether a telecom customer is likely to churn (leave the service). It uses an XGBoost classifier, model evaluation metrics, threshold tuning, feature importance, SHAP explainability, and a Streamlit interface for interactive predictions.
Project Overview
Customer churn can reduce recurring revenue and increase customer acquisition costs. This project explores customer data to identify patterns associated with churn and provides a risk estimate that can help a business prioritize retention reviews.
Important: Model predictions are estimates, not guarantees. Any retention action should be reviewed alongside customer context and business costs.
Features
- Exploratory data analysis (EDA) of telecom customer data
- Categorical preprocessing and numerical feature handling through a saved pipeline
- Model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix
- Experimental decision-threshold tuning
- XGBoost feature-importance visualization
- SHAP summary and individual-customer explanations
- Streamlit app for entering customer details and viewing churn probability
Model Results
The following results were reported during the project experiments on the held-out test set:
Metric	Logistic Regression	Random Forest	XGBoost
Accuracy	0.7381	0.7658	0.7984
Precision	0.5043	0.5521	0.6510
Recall	0.7834	0.6230	0.5187
F1-score	0.6136	0.5854	0.5774
ROC-AUC	0.8413	0.8218	0.8427


Threshold experiment
An experimental XGBoost classification threshold of 0.25 was selected to prioritize identifying more customers who may churn.
Metric	XGBoost at threshold 0.25
Accuracy	0.7452
Precision	0.5128
Recall	0.8048
F1-score	0.6264
ROC-AUC	0.8427


Confusion matrix at threshold 0.25:
	Predicted: No Churn	Predicted: Churn
Actual: No Churn	749	286
Actual: Churn	73	301


A lower threshold increased recall but also increased false positives. The threshold should be validated against business costs before production use. The ROC-AUC is threshold-independent, so it remains unchanged when only the decision threshold changes.
Key Findings
The model's feature-importance analysis ranked these features highly:
1. Month-to-month contract
2. Fiber-optic internet service
3. No online security
4. No tech support
5. DSL internet service
6. Two-year contract
7. Internet service unavailable
8. Electronic-check payment method
9. Customer tenure
Feature importance and SHAP values describe how the trained model uses features; they do not prove causation. SHAP explanations for an individual customer describe contributions to the model output for that prediction.
Project Structure
customer-churn-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── customer_churn_xgboost.joblib
├── notebooks/
│   └── churn_analysis.ipynb
└── reports/
    └── figures/
        ├── feature_importance.png
        └── shap_summary.png
Your actual filenames and folders may differ. Update this structure to match the files in your repository.
Dataset
This project is based on the IBM Telco Customer Churn dataset.
- Kaggle dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Download the dataset from Kaggle and place it in your local data folder if your notebook requires it. Do not commit raw data to GitHub unless you have checked its license and redistribution terms.
Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
Replace YOUR-USERNAME with your GitHub username.
2. Create and activate a virtual environment (recommended)
Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, you can run the environment's Python directly:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Run the Streamlit App
Ensure the saved model exists at:
models/customer_churn_xgboost.joblib
Then run:
python -m streamlit run app.py
Streamlit normally opens the app at http://localhost:8501.
The app collects customer attributes, passes them to the saved preprocessing/model pipeline, and displays churn probability and the predicted class using the configured threshold.
Input compatibility: The input column names, column order, and category labels in app.py must match the features used when training the saved model. If the app reports missing columns or invalid categories, align the form fields with the model's saved features and training data.
Visualizations
Feature Importance

SHAP Summary

Make sure these image files are saved in reports/figures/ and that shap_summary.png is generated from the actual model's SHAP values. Avoid using an illustrative recreation as if it were a model-generated result.
Technologies Used
- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Joblib
Limitations
- Results depend on the dataset, preprocessing, train/test split, and selected threshold.
- Precision and recall represent different business trade-offs; retention campaigns also have costs.
- Feature importance is not causal evidence.
- Validate the threshold on a separate validation set and monitor model performance over time before deployment.
- The saved Joblib artifact should only be loaded from a trusted source, and compatible package versions should be used.
Future Improvements
- Add validation-based threshold selection and cost-sensitive evaluation
- Add a lift chart and Precision-Recall curve
- Add a downloadable customer risk report
- Use SHAP to explain individual predictions in the app
- Add automated tests and package-version pinning
- Deploy the Streamlit app and monitor prediction quality
Author
Ali Muhammad
- GitHub: https://github.com/alimuhammad-007
- LinkedIn: https://www.linkedin.com/in/ali-muhammad-41b921372/
License
Choose a license before publishing if you want others to reuse or modify this project. For example, an MIT License can be added in a separate LICENSE file.
