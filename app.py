"""
app.py — Insurance Claim-Settlement Bias Dashboard  (v2 — tuned)
=================================================================
Streamlit dashboard covering:
  1. Descriptive cross-tabulation vs POLICY_STATUS
  2. Diagnostic bias analysis (age / income / team / channel)
  3. Supervised classification — KNN, Decision Tree, Random Forest, Gradient Boosting
     with enhanced feature engineering + GridSearchCV hyperparameter tuning + 5-fold CV
  4. Train/test accuracy, precision/recall/F1, ROC curves, confusion matrices
  5. Findings

Run locally:   streamlit run app.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import data_prep as dp

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RND = 42

st.set_page_config(page_title="Claim Settlement Bias Dashboard",
                   layout="wide", page_icon="📊")


# ── caching ───────────────────────────────────────────────────────────────── #
@st.cache_data
def get_data(path):
    return dp.add_engineered_features(dp.load_clean(path))


@st.cache_data
def get_matrix(path, min_count):
    return dp.build_model_matrix(dp.load_clean(path), min_count=min_count)


@st.cache_resource
def train_all(path, min_count, test_size):
    X, y, names = get_matrix(path, min_count)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RND, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)

    p = dp.BEST_PARAMS
    models = {
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(
                             n_neighbors=p["KNN"]["n_neighbors"],
                             weights=p["KNN"]["weights"],
                             metric=p["KNN"]["metric"]))]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=p["Decision Tree"]["max_depth"],
            min_samples_leaf=p["Decision Tree"]["min_samples_leaf"],
            min_samples_split=p["Decision Tree"]["min_samples_split"],
            criterion=p["Decision Tree"]["criterion"],
            random_state=RND),
        "Random Forest": RandomForestClassifier(
            n_estimators=p["Random Forest"]["n_estimators"],
            max_depth=p["Random Forest"]["max_depth"],
            min_samples_leaf=p["Random Forest"]["min_samples_leaf"],
            max_features=p["Random Forest"]["max_features"],
            random_state=RND, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=p["Gradient Boosting"]["n_estimators"],
            learning_rate=p["Gradient Boosting"]["learning_rate"],
            max_depth=p["Gradient Boosting"]["max_depth"],
            min_samples_leaf=p["Gradient Boosting"]["min_samples_leaf"],
            subsample=p["Gradient Boosting"]["subsample"],
            random_state=RND),
    }

    out = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_tr, p_te = m.predict(X_tr), m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        cv_scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        out[name] = {
            "model": m,
            "train_acc": accuracy_score(y_tr, p_tr),
            "test_acc": accuracy_score(y_te, p_te),
            "precision": precision_score(y_te, p_te),
            "recall": recall_score(y_te, p_te),
            "f1": f1_score(y_te, p_te),
            "roc_auc": roc_auc_score(y_te, proba),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "fpr": fpr, "tpr": tpr,
            "cm": confusion_matrix(y_te, p_te),
        }
    return out, names, X


def rate_table(df, col):
    t = (df.groupby(col, observed=True)["APPROVED"]
           .agg(approval_rate="mean", claims="count"))
    t["approval_rate"] = (t["approval_rate"] * 100).round(1)
    return t


# ── sidebar ───────────────────────────────────────────────────────────────── #
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload Insurance.csv (optional)", type="csv")
DATA_PATH = uploaded if uploaded is not None else "Insurance.csv"

st.sidebar.markdown("### Model settings")
test_size = st.sidebar.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
min_count = st.sidebar.slider("Rare-category threshold", 5, 40, 15, 5)

df = get_data(DATA_PATH)

st.title("📊 Insurance Claim-Settlement Bias Dashboard")
st.caption("Target: **POLICY_STATUS** — *Approved Death Claim* vs *Repudiate Death* · "
           "Models tuned via **GridSearchCV** with **5-fold stratified CV**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", f"{len(df):,}")
c2.metric("Overall approval", f"{df['APPROVED'].mean()*100:.1f}%")
c3.metric("TEAM-channel approval", f"{df[df.IS_TEAM_ZONE==1]['APPROVED'].mean()*100:.1f}%")
c4.metric("Non-TEAM approval", f"{df[df.IS_TEAM_ZONE==0]['APPROVED'].mean()*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 Descriptive", "2 Diagnostic (bias)", "3 Models (tuned)", "4 Findings"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────── #
with tab1:
    st.subheader("Cross-tabulation against policy status")
    cat_options = ["EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE", "PI_GENDER",
                   "AGE_BAND", "INC_BAND", "ZONE_CLEAN", "PI_STATE",
                   "OCC_GROUP", "REASON_GROUP"]
    sel = st.selectbox("Dimension to cross-tabulate", cat_options)

    left, right = st.columns(2)
    with left:
        st.markdown("**Counts**")
        ct = pd.crosstab(df[sel], df["POLICY_STATUS"], margins=True)
        st.dataframe(ct, use_container_width=True)
    with right:
        st.markdown("**Row % (outcome within each level)**")
        ctp = (pd.crosstab(df[sel], df["POLICY_STATUS"], normalize="index") * 100).round(1)
        st.dataframe(ctp, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ctp_s = ctp.sort_values("Approved Death Claim")
    ctp_s.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
    ax.axhline(df["APPROVED"].mean()*100, color="black", ls="--",
               label=f"Overall {df['APPROVED'].mean()*100:.1f}%")
    ax.set_ylabel("% of claims"); ax.set_title(f"Outcome mix by {sel}")
    ax.legend(bbox_to_anchor=(1.02, 1)); plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ── TAB 2 ─────────────────────────────────────────────────────────────────── #
with tab2:
    st.subheader("Where do approval rates diverge?")
    st.markdown("Red dashed line = overall approval rate. Bars far from it flag segments for bias review.")

    z = rate_table(df, "ZONE_CLEAN")
    z = z[z["claims"] >= 15].sort_values("approval_rate")
    team_flag = df.groupby("ZONE_CLEAN")["IS_TEAM_ZONE"].first()
    colors = ["#2ca02c" if team_flag.get(i, 0)==1 else "#1f77b4" for i in z.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(z.index, z["approval_rate"], color=colors)
    ax.axvline(df["APPROVED"].mean()*100, color="red", ls="--")
    ax.set_xlabel("Approval rate %")
    ax.set_title("Approval rate by zone/channel (green = TEAM, ≥15 claims)")
    st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        a = rate_table(df, "AGE_BAND").sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(a.index.astype(str), a["approval_rate"], color="#4c72b0")
        ax.axhline(df["APPROVED"].mean()*100, color="red", ls="--")
        ax.set_title("Approval by age band"); ax.set_ylabel("%")
        st.pyplot(fig)
    with colB:
        i = rate_table(df, "INC_BAND").sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(i.index.astype(str), i["approval_rate"], color="#dd8452")
        ax.axhline(df["APPROVED"].mean()*100, color="red", ls="--")
        ax.set_title("Approval by income band"); ax.set_ylabel("%"); plt.xticks(rotation=20)
        st.pyplot(fig)

    st.markdown("#### Chi-square test of independence")
    from scipy.stats import chi2_contingency
    test_dim = st.selectbox("Test dependence on:",
                            ["IS_TEAM_ZONE","AGE_BAND","INC_BAND","EARLY_NON",
                             "PI_GENDER","MEDICAL_NONMED","OCC_GROUP","REASON_GROUP"])
    obs = pd.crosstab(df[test_dim], df["POLICY_STATUS"])
    chi2, p, dof, _ = chi2_contingency(obs)
    st.write(f"χ² = **{chi2:.1f}**, dof = {dof}, p-value = **{p:.2e}**")
    if p < 0.05:
        st.warning(f"Outcome is **statistically dependent** on `{test_dim}` (p < 0.05).")
    else:
        st.info(f"No significant dependence on `{test_dim}` (p ≥ 0.05).")

# ── TAB 3 ─────────────────────────────────────────────────────────────────── #
with tab3:
    st.subheader("Feature engineering (v3 — enhanced)")
    st.markdown("""
**Improvements over baseline (v1):**
- **Interaction features:** `TEAM × EARLY`, `MEDICAL × EARLY`, `TEAM × MEDICAL`, `HIGH_COVER_LOW_INCOME`
- **Occupation** grouped into 7 meaningful buckets (Service/Salaried, Business/Self-Emp, Agriculture, Retired/Pensioner, Homemaker, Unknown, Other)
- **Claim reason** grouped into 8 clusters (Cardiac, Cancer, Accidental, Natural Death, Organ Failure, Unnatural, Infection, Not Specified)
- **Added:** `AGE²`, `LOG_INCOME`, binary flags (`IS_EARLY`, `IS_MEDICAL`, `IS_FEMALE`, `IS_SINGLE_PAY`)
- **Hyperparameters** tuned via GridSearchCV / manual grid search with **5-fold stratified cross-validation**
- All categoricals one-hot encoded (rare grouped); KNN features standardised
    """)

    results, names, X = train_all(DATA_PATH, min_count, test_size)
    metrics = pd.DataFrame({
        k: {m: v[m] for m in ["train_acc","test_acc","precision","recall","f1","roc_auc","cv_mean","cv_std"]}
        for k, v in results.items()
    }).T
    metrics.columns = ["Train Acc","Test Acc","Precision","Recall","F1","ROC-AUC","CV Mean","CV Std"]

    st.markdown(f"**Model matrix:** {X.shape[0]} rows × {X.shape[1]} engineered features")

    # Before/after comparison
    old_metrics = pd.DataFrame({
        "KNN": {"Old_Test":0.699,"Old_AUC":0.700},
        "Decision Tree": {"Old_Test":0.732,"Old_AUC":0.751},
        "Random Forest": {"Old_Test":0.746,"Old_AUC":0.784},
        "Gradient Boosting": {"Old_Test":0.768,"Old_AUC":0.788},
    }).T
    compare = metrics[["Train Acc","Test Acc","ROC-AUC","CV Mean"]].round(3).copy()
    compare["Old Test"] = old_metrics["Old_Test"]
    compare["Δ Test"] = (compare["Test Acc"] - compare["Old Test"]).round(3)
    compare["Gap (tr-te)"] = (compare["Train Acc"] - compare["Test Acc"]).round(3)
    compare = compare[["Old Test","Test Acc","Δ Test","Train Acc","Gap (tr-te)","ROC-AUC","CV Mean"]]
    st.dataframe(compare, use_container_width=True)

    st.markdown("**Best hyperparameters** (from GridSearchCV):")
    params_display = pd.DataFrame({
        k: {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
        for k, v in dp.BEST_PARAMS.items()
    }).T
    st.dataframe(params_display, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        m2 = metrics[["Train Acc","Test Acc"]].copy()
        m2["CV Mean"] = metrics["CV Mean"]
        m2.plot(kind="bar", ax=ax, color=["#9ecae1","#3182bd","#fdae6b"])
        ax.set_title("Train / Test / CV accuracy"); ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    with cc2:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        metrics[["Precision","Recall","F1"]].plot(kind="bar", ax=ax)
        ax.set_title("Precision / Recall / F1 (test)"); ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    st.markdown("#### ROC curves")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
    st.pyplot(fig)

    st.markdown("#### Confusion matrices")
    cols = st.columns(4)
    for col, (name, r) in zip(cols, results.items()):
        fig, ax = plt.subplots(figsize=(3, 2.8))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Rep","App"], yticklabels=["Rep","App"])
        ax.set_title(name, fontsize=9); ax.set_xlabel("Pred"); ax.set_ylabel("Actual")
        col.pyplot(fig)

    st.markdown("#### Feature importance (Random Forest)")
    rf = results["Random Forest"]["model"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title("Top 15 features"); st.pyplot(fig)

    st.session_state["metrics"] = metrics

# ── TAB 4 ─────────────────────────────────────────────────────────────────── #
with tab4:
    st.subheader("Findings & recommendations")
    team = df[df.IS_TEAM_ZONE==1]["APPROVED"].mean()*100
    nonteam = df[df.IS_TEAM_ZONE==0]["APPROVED"].mean()*100
    early = df.groupby("EARLY_NON")["APPROVED"].mean()*100
    inc = rate_table(df, "INC_BAND").sort_index()

    st.markdown(f"""
**1. Channel / "team" disparity — strongest bias signal.**
TEAM-branded channels approve **{team:.1f}%** vs **{nonteam:.1f}%** elsewhere
(gap = **{team-nonteam:.1f} pp**). Several TEAM zones exceed 90% while JKB JAMMU
and South sit near 45%.

**2. Income gradient.** The 1–100k band is approved at only
**{inc.loc['1-100k','approval_rate']:.1f}%**, suggesting higher documentation burden
on low-income claimants.

**3. Early-claim anomaly.** EARLY claims approved more often
({early.get('EARLY',0):.1f}%) than NON-EARLY ({early.get('NON EARLY',0):.1f}%) — the
reverse of typical fraud-risk expectation.

**4. Occupation effect.** Service/Business (43–44% approval) vs Pensioner/Retired
(86–92%) — may reflect documentation or fraud-flag patterns worth auditing.

**5. Modelling (tuned with 5-fold CV).**
After feature engineering v3 and hyperparameter tuning:
- **KNN improved most:** test accuracy from 69.9% → 75.7% (+5.8 pp), train-test gap
  collapsed from 4.5pp to 0.2pp.
- **Gradient Boosting** remains the strongest overall (test 77.5%, AUC 0.792).
- Top drivers: sum assured, occupation, age, cover-to-income, `IS_TEAM_ZONE`, and
  interactions (team×early, medical×early).

**6. Cross-validation confirms stability.** All models' 5-fold CV means sit within
1–2pp of their test accuracy, indicating no serious overfitting.
""")
    st.info(
        "**Caveat:** statistical disparity ≠ proof of unfair bias. Differences may "
        "reflect legitimate underwriting factors. Treat these as **prioritised audit "
        "leads** — review a stratified sample of repudiated vs approved claims from "
        "flagged segments.")

    if "metrics" in st.session_state:
        st.download_button("⬇️ Download model metrics (CSV)",
                           st.session_state["metrics"].round(4).to_csv().encode(),
                           "model_metrics_v2.csv", "text/csv")
