# 🔍 Insurance Claim Bias Audit Dashboard

A Streamlit dashboard for auditing bias in insurance claim settlement processes.

## Features
- **Descriptive Analysis**: Cross-tabulations of all variables vs Policy Status
- **Diagnostic Analysis**: Age, income, zone/team-level bias detection with heatmaps
- **ML Models**: KNN, Decision Tree, Random Forest, Gradient Boosting with feature engineering
- **Model Evaluation**: Train/test accuracy, precision/recall/F1, ROC curves, confusion matrices
- **Findings**: Automated bias findings and recommendations

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Fork/upload this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click Deploy!

> **Note**: Make sure `Insurance.csv` is in the same folder as `app.py`.

## Dataset Columns
| Column | Description |
|--------|-------------|
| POLICY_NO | Policy number |
| PI_NAME | Insured person name |
| PI_GENDER | Gender |
| SUM_ASSURED | Insured sum |
| ZONE | Sales/settlement zone |
| PAYMENT_MODE | Premium payment frequency |
| EARLY_NON | Whether claim is early |
| PI_OCCUPATION | Occupation |
| MEDICAL_NONMED | Medical/Non-medical policy |
| PI_STATE | State of the insured |
| REASON_FOR_CLAIM | Claim reason |
| PI_AGE | Age of insured |
| PI_ANNUAL_INCOME | Annual income |
| POLICY_STATUS | **Target**: Approved / Repudiated |
