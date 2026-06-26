# 📊 Insurance Claim-Settlement Bias Dashboard

An end-to-end Streamlit app that audits a life-insurance death-claim dataset for
possible bias in the settlement decision (`POLICY_STATUS`: *Approved Death Claim*
vs *Repudiate Death*).

It delivers all five objectives:

1. **Descriptive analysis** – cross-tabulations of every dimension against policy status.
2. **Diagnostic analysis** – approval-rate disparities by age, income, channel/"team",
   plus a Chi-square independence test.
3. **Supervised classification** – KNN, Decision Tree, Random Forest and Gradient
   Boosting, with full feature engineering.
4. **Evaluation** – train/test accuracy, precision/recall/F1, ROC curves and
   confusion matrices for every model, plus feature importance.
5. **Findings** – plain-language summary and audit recommendations.

## 🗂 Project structure

```
insurance_bias/
├── app.py              # Streamlit dashboard (5 tabs)
├── data_prep.py        # shared cleaning + feature engineering
├── run_analysis.py     # static run -> writes plots to outputs/
├── Insurance.csv       # dataset
├── requirements.txt
├── FINDINGS.md         # written findings
└── outputs/            # PNG charts + metrics from a static run
```

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

To regenerate the static charts in `outputs/`:

```bash
python run_analysis.py
```

## ☁️ Deploy on Streamlit Community Cloud

1. Create a **new GitHub repository** and push every file in this folder
   (keep `app.py`, `data_prep.py`, `requirements.txt` and `Insurance.csv` at the repo root):

   ```bash
   git init
   git add .
   git commit -m "Insurance claim-settlement bias dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **https://share.streamlit.io**, sign in with GitHub, click **New app**.
3. Pick your repo / branch (`main`) and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

The sidebar lets you upload a different `Insurance.csv` and tune model settings
(test-size, KNN k, tree depth, forest size, rare-category threshold) live.

## ⚠️ Interpretation note

Statistical disparity is **not by itself** proof of unfair bias — it can reflect
legitimate underwriting factors (medical evidence, documentation, policy vintage).
Treat the flagged segments as **prioritised audit leads**, then review a stratified
sample of repudiated claims against comparable approved ones.
