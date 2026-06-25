"""
app.py — Insurance Claim-Settlement Bias Dashboard
==================================================
Streamlit dashboard covering:
  1. Descriptive cross-tabulation vs POLICY_STATUS
  2. Diagnostic bias analysis (age / income / team / channel)
  3. Supervised classification (KNN, Decision Tree, Random Forest, Gradient Boosting)
     with feature engineering
  4. Train/test accuracy, precision/recall/F1, ROC curves, confusion matrices
  5. Findings

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> point at app.py
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
from sklearn.model_selection import train_test_split
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


# --------------------------------------------------------------------------- #
# Data + model caching
# --------------------------------------------------------------------------- #
@st.cache_data
def get_data(path):
    return dp.add_engineered_features(dp.load_clean(path))


@st.cache_data
def get_matrix(path, min_count):
    df = dp.load_clean(path)
    return dp.build_model_matrix(df, min_count=min_count)


@st.cache_resource
def train_all(path, min_count, test_size, knn_k, tree_depth, rf_trees):
    X, y, names = get_matrix(path, min_count)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RND, stratify=y)
    models = {
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=knn_k))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth,
                                                min_samples_leaf=20, random_state=RND),
        "Random Forest": RandomForestClassifier(n_estimators=rf_trees, min_samples_leaf=5,
                                                n_jobs=-1, random_state=RND),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RND),
    }
    out = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_tr, p_te = m.predict(X_tr), m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        out[name] = {
            "model": m,
            "train_acc": accuracy_score(y_tr, p_tr),
            "test_acc": accuracy_score(y_te, p_te),
            "precision": precision_score(y_te, p_te),
            "recall": recall_score(y_te, p_te),
            "f1": f1_score(y_te, p_te),
            "roc_auc": roc_auc_score(y_te, proba),
            "fpr": fpr, "tpr": tpr,
            "cm": confusion_matrix(y_te, p_te),
        }
    return out, names, X


def rate_table(df, col):
    t = (df.groupby(col, observed=True)["APPROVED"]
           .agg(approval_rate="mean", claims="count"))
    t["approval_rate"] = (t["approval_rate"] * 100).round(1)
    return t


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload Insurance.csv (optional)", type="csv")
DATA_PATH = uploaded if uploaded is not None else "Insurance.csv"

st.sidebar.markdown("### Model settings")
test_size = st.sidebar.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
min_count = st.sidebar.slider("Rare-category threshold", 5, 40, 15, 5)
knn_k = st.sidebar.slider("KNN neighbours (k)", 3, 41, 15, 2)
tree_depth = st.sidebar.slider("Decision-tree max depth", 2, 15, 6, 1)
rf_trees = st.sidebar.slider("Random-forest trees", 100, 600, 400, 50)

df = get_data(DATA_PATH)

st.title("📊 Insurance Claim-Settlement Bias Dashboard")
st.caption("Target variable: **POLICY_STATUS** — *Approved Death Claim* vs *Repudiate Death*")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", f"{len(df):,}")
c2.metric("Overall approval", f"{df['APPROVED'].mean()*100:.1f}%")
c3.metric("TEAM-channel approval", f"{df[df.IS_TEAM_ZONE==1]['APPROVED'].mean()*100:.1f}%")
c4.metric("Non-TEAM approval", f"{df[df.IS_TEAM_ZONE==0]['APPROVED'].mean()*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 Descriptive (cross-tabs)", "2 Diagnostic (bias)",
     "3 Models", "4 Findings"])

# --------------------------------------------------------------------------- #
# TAB 1 — Descriptive cross-tabulation
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Cross-tabulation against policy status")
    cat_options = ["EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE", "PI_GENDER",
                   "AGE_BAND", "INC_BAND", "ZONE_CLEAN", "PI_STATE"]
    sel = st.selectbox("Choose a dimension to cross-tabulate", cat_options)

    left, right = st.columns(2)
    with left:
        st.markdown("**Counts**")
        ct = pd.crosstab(df[sel], df["POLICY_STATUS"], margins=True)
        st.dataframe(ct, use_container_width=True)
    with right:
        st.markdown("**Row % (approval / repudiation within each level)**")
        ctp = (pd.crosstab(df[sel], df["POLICY_STATUS"], normalize="index") * 100).round(1)
        st.dataframe(ctp, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ctp_sorted = ctp.sort_values("Approved Death Claim")
    ctp_sorted.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
    ax.axhline(df["APPROVED"].mean() * 100, color="black", ls="--",
               label=f"Overall approval {df['APPROVED'].mean()*100:.1f}%")
    ax.set_ylabel("% of claims")
    ax.set_title(f"Outcome mix by {sel}")
    ax.legend(bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --------------------------------------------------------------------------- #
# TAB 2 — Diagnostic bias analysis
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Where do approval rates diverge?")
    st.markdown(
        "The red dashed line is the **overall approval rate**. Bars far from it "
        "flag segments treated differently — the candidates for bias review.")

    # Zone / channel
    z = rate_table(df, "ZONE_CLEAN")
    z = z[z["claims"] >= 15].sort_values("approval_rate")
    team_flag = df.groupby("ZONE_CLEAN")["IS_TEAM_ZONE"].first()
    colors = ["#2ca02c" if team_flag.get(i, 0) == 1 else "#1f77b4" for i in z.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(z.index, z["approval_rate"], color=colors)
    ax.axvline(df["APPROVED"].mean() * 100, color="red", ls="--")
    ax.set_xlabel("Approval rate %")
    ax.set_title("Approval rate by zone/channel (green = TEAM channel, ≥15 claims)")
    st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        a = rate_table(df, "AGE_BAND").sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(a.index.astype(str), a["approval_rate"], color="#4c72b0")
        ax.axhline(df["APPROVED"].mean() * 100, color="red", ls="--")
        ax.set_title("Approval by age band"); ax.set_ylabel("%")
        st.pyplot(fig)
    with colB:
        i = rate_table(df, "INC_BAND").sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(i.index.astype(str), i["approval_rate"], color="#dd8452")
        ax.axhline(df["APPROVED"].mean() * 100, color="red", ls="--")
        ax.set_title("Approval by income band"); ax.set_ylabel("%")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    st.markdown("#### Statistical test (Chi-square of independence)")
    from scipy.stats import chi2_contingency
    test_dim = st.selectbox("Test whether outcome depends on:",
                            ["IS_TEAM_ZONE", "AGE_BAND", "INC_BAND",
                             "EARLY_NON", "PI_GENDER", "MEDICAL_NONMED"], key="chi")
    obs = pd.crosstab(df[test_dim], df["POLICY_STATUS"])
    chi2, p, dofree, _ = chi2_contingency(obs)
    st.write(f"χ² = **{chi2:.1f}**, dof = {dofree}, p-value = **{p:.2e}**")
    if p < 0.05:
        st.warning(f"Outcome is **statistically dependent** on `{test_dim}` "
                   f"(p < 0.05). Disparity is unlikely to be random chance.")
    else:
        st.info(f"No statistically significant dependence on `{test_dim}` (p ≥ 0.05).")

# --------------------------------------------------------------------------- #
# TAB 3 — Models
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Feature engineering")
    st.markdown(
        "- Money fields parsed to numeric; `SUM_ASSURED` log-transformed for KNN\n"
        "- `IS_TEAM_ZONE`, `INCOME_DECLARED`, `HAS_CLAIM_REASON`, `COVER_TO_INCOME` engineered\n"
        "- Zone casing duplicates merged; rare categories grouped to `OTHER`\n"
        "- All categoricals one-hot encoded; KNN features standardised")

    results, names, X = train_all(DATA_PATH, min_count, test_size,
                                  knn_k, tree_depth, rf_trees)
    metrics = pd.DataFrame({k: {m: v[m] for m in
                                ["train_acc", "test_acc", "precision", "recall", "f1", "roc_auc"]}
                            for k, v in results.items()}).T
    st.markdown(f"**Model matrix:** {X.shape[0]} rows × {X.shape[1]} engineered features")
    st.dataframe(metrics.round(3), use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        metrics[["train_acc", "test_acc"]].plot(kind="bar", ax=ax,
                                                color=["#9ecae1", "#3182bd"])
        ax.set_title("Train vs Test accuracy"); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    with cc2:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        metrics[["precision", "recall", "f1"]].plot(kind="bar", ax=ax)
        ax.set_title("Precision / Recall / F1"); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    st.markdown("#### ROC curves (model stability)")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.markdown("#### Confusion matrices")
    cols = st.columns(4)
    for col, (name, r) in zip(cols, results.items()):
        fig, ax = plt.subplots(figsize=(3, 2.8))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Rep", "App"], yticklabels=["Rep", "App"])
        ax.set_title(name, fontsize=9); ax.set_xlabel("Pred"); ax.set_ylabel("Actual")
        col.pyplot(fig)

    st.markdown("#### What drives the model? (Random Forest importance)")
    rf = results["Random Forest"]["model"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title("Top 15 features")
    st.pyplot(fig)
    st.session_state["imp"] = imp
    st.session_state["metrics"] = metrics

# --------------------------------------------------------------------------- #
# TAB 4 — Findings
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Findings & recommendations")
    team = df[df.IS_TEAM_ZONE == 1]["APPROVED"].mean() * 100
    nonteam = df[df.IS_TEAM_ZONE == 0]["APPROVED"].mean() * 100
    early = df.groupby("EARLY_NON")["APPROVED"].mean() * 100
    inc = rate_table(df, "INC_BAND").sort_index()

    st.markdown(f"""
**Channel / "team" disparity is the strongest signal.**
TEAM-branded channels approve **{team:.1f}%** of death claims versus **{nonteam:.1f}%**
elsewhere — a gap of **{team-nonteam:.1f} points**. Several TEAM zones sit above 90%
while JKB JAMMU and South sit near 45%.

**Income gradient.** The lowest income band (1–100k) is approved at only
**{inc.loc['1-100k','approval_rate']:.1f}%**, well below higher bands — worth checking
whether documentation burden falls unevenly on low-income claimants.

**Early-claim counter-intuition.** EARLY claims are approved **more** often
({early.get('EARLY',0):.1f}%) than NON-EARLY ({early.get('NON EARLY',0):.1f}%),
the opposite of the usual fraud-risk expectation — a process anomaly to probe.

**Modelling.** Gradient Boosting / Random Forest reach the best ROC-AUC. Channel
(`IS_TEAM_ZONE`), occupation, age, sum-assured and cover-to-income all carry
predictive weight, confirming the descriptive disparities are systematic rather
than noise.
""")
    st.info(
        "**Important caveat:** statistical disparity is *not by itself* proof of "
        "unfair bias. Differences may partly reflect legitimate underwriting factors "
        "(documentation quality, medical evidence, policy vintage). Treat these as "
        "**prioritised audit leads** — pull a stratified sample of repudiated claims "
        "from the low-approval zones/segments and review them against approved peers.")

    if "metrics" in st.session_state:
        st.download_button("⬇️ Download model metrics (CSV)",
                           st.session_state["metrics"].round(4).to_csv().encode(),
                           "model_metrics.csv", "text/csv")
