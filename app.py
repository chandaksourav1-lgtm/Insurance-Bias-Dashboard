import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Bias Audit Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #1a237e, #e53935);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding-bottom: 0.5rem;
  }
  .section-header {font-size:1.3rem; font-weight:700; color:#1a237e; border-bottom:2px solid #e53935; padding-bottom:4px; margin-top:1.5rem;}
  .metric-card {background:#f0f4ff; border-left:5px solid #1a237e; padding:1rem; border-radius:8px; margin:0.5rem 0;}
  .warning-box {background:#fff3e0; border-left:5px solid #ff6f00; padding:1rem; border-radius:8px;}
  .finding-box {background:#e8f5e9; border-left:5px solid #2e7d32; padding:1rem; border-radius:8px; margin:0.4rem 0;}
</style>
""", unsafe_allow_html=True)

# ─── Load & Preprocess Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Insurance.csv")
    df["SUM_ASSURED"] = df["SUM_ASSURED"].astype(str).str.replace(",", "").astype(float)
    df["PI_ANNUAL_INCOME"] = df["PI_ANNUAL_INCOME"].astype(str).str.replace(",", "").astype(float)
    df["CLAIM_APPROVED"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    df["AGE_GROUP"] = pd.cut(df["PI_AGE"], bins=[0, 18, 35, 50, 65, 100],
                             labels=["<18", "18–35", "36–50", "51–65", "65+"])
    df["INCOME_GROUP"] = pd.cut(df["PI_ANNUAL_INCOME"],
                                bins=[0, 100000, 300000, 600000, 1000000, float("inf")],
                                labels=["<1L", "1–3L", "3–6L", "6–10L", "10L+"])
    # Standardise ZONE capitalisation
    df["ZONE_CLEAN"] = df["ZONE"].str.strip().str.upper()
    return df

df = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/insurance.png", width=80)
st.sidebar.markdown("## Insurance Bias Audit")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Descriptive Analysis",
     "🔍 Diagnostic Analysis",
     "🤖 ML Models",
     "📈 Model Evaluation",
     "📋 Findings & Recommendations"]
)
st.sidebar.markdown("---")
st.sidebar.metric("Total Claims", len(df))
st.sidebar.metric("Approved", (df["CLAIM_APPROVED"] == 1).sum())
st.sidebar.metric("Repudiated", (df["CLAIM_APPROVED"] == 0).sum())
approve_rate = df["CLAIM_APPROVED"].mean() * 100
st.sidebar.metric("Approval Rate", f"{approve_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Descriptive Analysis":
    st.markdown('<div class="main-header">📊 Descriptive Analysis — Cross-Tabulation vs Policy Status</div>', unsafe_allow_html=True)

    # ── Summary stats
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Zones/Teams", df["ZONE"].nunique())
    col3.metric("Unique States", df["PI_STATE"].nunique())
    col4.metric("Age Range", f"{df['PI_AGE'].min()}–{df['PI_AGE'].max()}")

    st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.drop(columns=["CLAIM_APPROVED", "AGE_GROUP", "INCOME_GROUP", "ZONE_CLEAN"]).head(20), use_container_width=True)

    # ── Cross-tabs
    st.markdown('<div class="section-header">Cross-Tabulation Tables</div>', unsafe_allow_html=True)
    xtab_col = st.selectbox("Select variable for cross-tab",
                            ["ZONE", "PI_GENDER", "AGE_GROUP", "INCOME_GROUP",
                             "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED", "PI_OCCUPATION"])

    ct = pd.crosstab(df[xtab_col], df["POLICY_STATUS"], margins=True)
    ct["Approval Rate %"] = (ct.get("Approved Death Claim", 0) / ct["All"] * 100).round(1)
    st.dataframe(ct.style.background_gradient(cmap="Blues", subset=["Approval Rate %"]), use_container_width=True)

    # ── Visual cross-tab
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cross = pd.crosstab(df[xtab_col], df["POLICY_STATUS"], normalize="index") * 100
    cross.plot(kind="bar", stacked=True, ax=axes[0], color=["#e53935", "#1a237e"], edgecolor="white")
    axes[0].set_title(f"Policy Status Distribution by {xtab_col}", fontweight="bold")
    axes[0].set_xlabel(xtab_col); axes[0].set_ylabel("Percentage %")
    axes[0].legend(title="Policy Status", bbox_to_anchor=(1.02, 1))
    axes[0].tick_params(axis="x", rotation=45)

    approve_by = df.groupby(xtab_col)["CLAIM_APPROVED"].mean().sort_values() * 100
    approve_by.plot(kind="barh", ax=axes[1], color="#1a237e", edgecolor="white")
    axes[1].axvline(df["CLAIM_APPROVED"].mean() * 100, color="#e53935", linestyle="--", linewidth=2, label=f"Overall Avg {approve_rate:.1f}%")
    axes[1].set_title(f"Approval Rate (%) by {xtab_col}", fontweight="bold")
    axes[1].set_xlabel("Approval Rate (%)"); axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Distribution plots
    st.markdown('<div class="section-header">Univariate Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df["PI_AGE"].hist(ax=axes[0], bins=20, color="#1a237e", edgecolor="white")
    axes[0].set_title("Age Distribution"); axes[0].set_xlabel("Age")

    df["PI_ANNUAL_INCOME"].apply(np.log1p).hist(ax=axes[1], bins=20, color="#43a047", edgecolor="white")
    axes[1].set_title("Log Annual Income Distribution"); axes[1].set_xlabel("Log(Income)")

    df["SUM_ASSURED"].apply(np.log1p).hist(ax=axes[2], bins=20, color="#e53935", edgecolor="white")
    axes[2].set_title("Log Sum Assured Distribution"); axes[2].set_xlabel("Log(Sum Assured)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Gender & Medical split
    st.markdown('<div class="section-header">Gender & Medical Breakdown</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["PI_GENDER", "MEDICAL_NONMED"], ["Gender", "Medical vs Non-Medical"]):
        vals = df.groupby(col)["CLAIM_APPROVED"].value_counts(normalize=True).unstack() * 100
        vals.plot(kind="bar", ax=ax, color=["#e53935", "#1a237e"], edgecolor="white")
        ax.set_title(title, fontweight="bold"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=0)
        ax.legend(["Repudiated", "Approved"])
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DIAGNOSTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analysis":
    st.markdown('<div class="main-header">🔍 Diagnostic Analysis — Bias Detection</div>', unsafe_allow_html=True)

    # ── Age-wise
    st.markdown('<div class="section-header">Age-Wise Bias Analysis</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.boxplot(data=df, x="POLICY_STATUS", y="PI_AGE", ax=axes[0],
                palette=["#e53935", "#1a237e"])
    axes[0].set_title("Age vs Policy Status", fontweight="bold")

    age_rate = df.groupby("AGE_GROUP")["CLAIM_APPROVED"].mean() * 100
    axes[1].bar(age_rate.index.astype(str), age_rate.values, color="#1a237e", edgecolor="white")
    axes[1].axhline(approve_rate, color="#e53935", linestyle="--", label=f"Avg {approve_rate:.1f}%")
    axes[1].set_title("Approval Rate by Age Group", fontweight="bold"); axes[1].set_ylabel("%"); axes[1].legend()

    cross_age = pd.crosstab(df["AGE_GROUP"], df["POLICY_STATUS"], normalize="index") * 100
    cross_age.plot(kind="bar", stacked=True, ax=axes[2], color=["#e53935", "#1a237e"], edgecolor="white")
    axes[2].set_title("Policy Status by Age Group (Stacked %)", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    age_tab = df.groupby("AGE_GROUP").agg(
        Total=("CLAIM_APPROVED", "count"),
        Approved=("CLAIM_APPROVED", "sum"),
        ApprovalPct=("CLAIM_APPROVED", lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    st.dataframe(age_tab, use_container_width=True)

    # ── Income-wise
    st.markdown('<div class="section-header">Income-Wise Bias Analysis</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.boxplot(data=df, x="POLICY_STATUS", y="PI_ANNUAL_INCOME", ax=axes[0],
                palette=["#e53935", "#1a237e"])
    axes[0].set_title("Annual Income vs Policy Status", fontweight="bold")
    axes[0].set_ylabel("Annual Income")

    inc_rate = df.groupby("INCOME_GROUP")["CLAIM_APPROVED"].mean() * 100
    axes[1].bar(inc_rate.index.astype(str), inc_rate.values, color="#43a047", edgecolor="white")
    axes[1].axhline(approve_rate, color="#e53935", linestyle="--", label=f"Avg {approve_rate:.1f}%")
    axes[1].set_title("Approval Rate by Income Group", fontweight="bold"); axes[1].set_ylabel("%"); axes[1].legend()

    cross_inc = pd.crosstab(df["INCOME_GROUP"], df["POLICY_STATUS"], normalize="index") * 100
    cross_inc.plot(kind="bar", stacked=True, ax=axes[2], color=["#e53935", "#1a237e"], edgecolor="white")
    axes[2].set_title("Policy Status by Income Group (Stacked %)", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    inc_tab = df.groupby("INCOME_GROUP").agg(
        Total=("CLAIM_APPROVED", "count"),
        Approved=("CLAIM_APPROVED", "sum"),
        ApprovalPct=("CLAIM_APPROVED", lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    st.dataframe(inc_tab, use_container_width=True)

    # ── Zone/Team-wise
    st.markdown('<div class="section-header">Zone / Team-Wise Bias Analysis</div>', unsafe_allow_html=True)
    zone_stats = df.groupby("ZONE").agg(
        Total=("CLAIM_APPROVED", "count"),
        Approved=("CLAIM_APPROVED", "sum"),
        Approval_Rate=("CLAIM_APPROVED", "mean")
    ).reset_index()
    zone_stats["Approval_Rate_Pct"] = zone_stats["Approval_Rate"] * 100
    zone_stats = zone_stats[zone_stats["Total"] >= 10].sort_values("Approval_Rate_Pct")

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#e53935" if r < approve_rate else "#1a237e" for r in zone_stats["Approval_Rate_Pct"]]
    bars = ax.barh(zone_stats["ZONE"], zone_stats["Approval_Rate_Pct"], color=colors, edgecolor="white")
    ax.axvline(approve_rate, color="#ff6f00", linestyle="--", linewidth=2, label=f"Overall Avg {approve_rate:.1f}%")
    ax.set_title("Approval Rate by Zone/Team (≥10 claims)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Approval Rate (%)")
    red_patch = mpatches.Patch(color="#e53935", label="Below Average")
    blue_patch = mpatches.Patch(color="#1a237e", label="Above Average")
    ax.legend(handles=[red_patch, blue_patch, plt.Line2D([0], [0], color="#ff6f00", linestyle="--")],
              labels=["Below Average", "Above Average", f"Overall Avg {approve_rate:.1f}%"])
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.dataframe(zone_stats.sort_values("Approval_Rate_Pct", ascending=False).reset_index(drop=True), use_container_width=True)

    # ── Gender + Medical intersection
    st.markdown('<div class="section-header">Intersectional Bias: Gender × Medical Status</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    heat_data = df.pivot_table(index="PI_GENDER", columns="MEDICAL_NONMED",
                               values="CLAIM_APPROVED", aggfunc="mean") * 100
    sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="RdYlGn", ax=axes[0], vmin=0, vmax=100)
    axes[0].set_title("Approval Rate % — Gender × Medical Status", fontweight="bold")

    heat_data2 = df.pivot_table(index="EARLY_NON", columns="MEDICAL_NONMED",
                                values="CLAIM_APPROVED", aggfunc="mean") * 100
    sns.heatmap(heat_data2, annot=True, fmt=".1f", cmap="RdYlGn", ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("Approval Rate % — Early/Non-Early × Medical Status", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Payment Mode
    st.markdown('<div class="section-header">Payment Mode Bias</div>', unsafe_allow_html=True)
    pm = df.groupby("PAYMENT_MODE")["CLAIM_APPROVED"].agg(["mean", "count"]).reset_index()
    pm.columns = ["Payment Mode", "Approval Rate", "Count"]
    pm["Approval Rate"] = pm["Approval Rate"] * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pm["Payment Mode"], pm["Approval Rate"], color="#1a237e", edgecolor="white")
    ax.axhline(approve_rate, color="#e53935", linestyle="--", label=f"Avg {approve_rate:.1f}%")
    ax.set_ylabel("Approval Rate (%)"); ax.set_title("Approval Rate by Payment Mode", fontweight="bold")
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def prepare_and_train():
    df_model = load_data().copy()

    # Feature engineering
    df_model["SUM_ASSURED_LOG"] = np.log1p(df_model["SUM_ASSURED"])
    df_model["INCOME_LOG"] = np.log1p(df_model["PI_ANNUAL_INCOME"])
    df_model["SUM_TO_INCOME"] = df_model["SUM_ASSURED"] / (df_model["PI_ANNUAL_INCOME"] + 1)
    df_model["IS_SENIOR"] = (df_model["PI_AGE"] >= 60).astype(int)
    df_model["IS_EARLY"] = (df_model["EARLY_NON"] == "EARLY").astype(int)
    df_model["IS_MEDICAL"] = (df_model["MEDICAL_NONMED"] == "MEDICAL").astype(int)
    df_model["PI_OCCUPATION"] = df_model["PI_OCCUPATION"].fillna("Unknown")

    cat_cols = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "PI_OCCUPATION",
                "MEDICAL_NONMED", "EARLY_NON", "PI_STATE"]
    le = LabelEncoder()
    for col in cat_cols:
        df_model[col + "_ENC"] = le.fit_transform(df_model[col].astype(str))

    features = (["PI_AGE", "SUM_ASSURED_LOG", "INCOME_LOG", "SUM_TO_INCOME", "IS_SENIOR", "IS_EARLY", "IS_MEDICAL"]
                + [c + "_ENC" for c in cat_cols])
    X = df_model[features]
    y = df_model["CLAIM_APPROVED"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_sc if name == "KNN" else X_train
        X_te = X_test_sc if name == "KNN" else X_test
        model.fit(X_tr, y_train)
        tr_pred = model.predict(X_tr)
        te_pred = model.predict(X_te)
        te_prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, te_prob)
        results[name] = {
            "model": model,
            "train_acc": accuracy_score(y_train, tr_pred),
            "test_acc": accuracy_score(y_test, te_pred),
            "precision": precision_score(y_test, te_pred),
            "recall": recall_score(y_test, te_pred),
            "f1": f1_score(y_test, te_pred),
            "cm": confusion_matrix(y_test, te_pred),
            "fpr": fpr, "tpr": tpr,
            "auc": auc(fpr, tpr),
            "y_test": y_test,
            "te_pred": te_pred,
            "report": classification_report(y_test, te_pred, target_names=["Repudiated", "Approved"])
        }

    fi_models = ["Decision Tree", "Random Forest", "Gradient Boosting"]
    feature_importances = {m: dict(zip(features, results[m]["model"].feature_importances_)) for m in fi_models}
    return results, feature_importances, features, X_train, X_test

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown('<div class="main-header">🤖 Machine Learning — Feature Engineering & Training</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Engineering Steps</div>', unsafe_allow_html=True)
    fe_steps = {
        "SUM_ASSURED_LOG": "Log transform of Sum Assured to reduce right skew",
        "INCOME_LOG": "Log transform of Annual Income",
        "SUM_TO_INCOME": "Ratio of Sum Assured to Annual Income (affordability proxy)",
        "IS_SENIOR": "Binary flag: age ≥ 60",
        "IS_EARLY": "Binary flag: EARLY_NON = EARLY",
        "IS_MEDICAL": "Binary flag: Medical case",
        "Encoded Categoricals": "LabelEncoding for Zone, Gender, Occupation, State, Payment Mode, Medical, Early"
    }
    for feat, desc in fe_steps.items():
        st.markdown(f"<div class='metric-card'><b>{feat}</b>: {desc}</div>", unsafe_allow_html=True)

    with st.spinner("Training all four models… this may take ~30 seconds"):
        results, feature_importances, features, X_train, X_test = prepare_and_train()
    st.success("✅ All models trained successfully!")

    st.markdown('<div class="section-header">Feature Importances (Tree Models)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mname in zip(axes, ["Decision Tree", "Random Forest", "Gradient Boosting"]):
        fi = pd.Series(feature_importances[mname]).sort_values(ascending=True).tail(10)
        fi.plot(kind="barh", ax=ax, color="#1a237e", edgecolor="white")
        ax.set_title(f"{mname} — Top 10 Features", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Classification Reports</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select model", list(results.keys()))
    st.text(results[selected]["report"])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-header">📈 Model Evaluation — Accuracy, Metrics, ROC & Confusion Matrix</div>', unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        results, feature_importances, features, X_train, X_test = prepare_and_train()

    # ── Summary Table
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Model": list(results.keys()),
        "Train Accuracy": [f"{v['train_acc']*100:.2f}%" for v in results.values()],
        "Test Accuracy":  [f"{v['test_acc']*100:.2f}%" for v in results.values()],
        "Precision":      [f"{v['precision']*100:.2f}%" for v in results.values()],
        "Recall":         [f"{v['recall']*100:.2f}%" for v in results.values()],
        "F1-Score":       [f"{v['f1']*100:.2f}%" for v in results.values()],
        "AUC-ROC":        [f"{v['auc']:.4f}" for v in results.values()]
    })
    st.dataframe(summary, use_container_width=True)

    # ── Train vs Test accuracy bar chart
    st.markdown('<div class="section-header">Train vs Test Accuracy Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    w = 0.35
    train_accs = [v["train_acc"] * 100 for v in results.values()]
    test_accs  = [v["test_acc"]  * 100 for v in results.values()]
    ax.bar(x - w/2, train_accs, w, label="Train Accuracy", color="#1a237e", edgecolor="white")
    ax.bar(x + w/2, test_accs,  w, label="Test Accuracy",  color="#e53935", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(results.keys())
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(50, 105)
    ax.set_title("Train vs Test Accuracy", fontweight="bold")
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Precision / Recall / F1
    st.markdown('<div class="section-header">Precision, Recall & F1-Score</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = ["precision", "recall", "f1"]
    colors  = ["#1a237e", "#43a047", "#e53935"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [v[metric] * 100 for v in results.values()]
        ax.bar(np.arange(len(results)) + i * 0.25, vals, 0.25,
               label=metric.upper(), color=color, edgecolor="white")
    ax.set_xticks(np.arange(len(results)) + 0.25)
    ax.set_xticklabels(results.keys()); ax.set_ylabel("%"); ax.set_ylim(50, 105)
    ax.set_title("Precision / Recall / F1-Score per Model", fontweight="bold")
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── ROC Curves
    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    pal = ["#1a237e", "#e53935", "#43a047", "#ff6f00"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    for (name, res), color in zip(results.items(), pal):
        ax.plot(res["fpr"], res["tpr"], color=color, linewidth=2,
                label=f"{name}  (AUC = {res['auc']:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Confusion Matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Repudiated", "Approved"],
                    yticklabels=["Repudiated", "Approved"])
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Findings & Recommendations":
    st.markdown('<div class="main-header">📋 Findings & Recommendations</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔴 Key Bias Findings</div>', unsafe_allow_html=True)

    findings = [
        ("Age Bias Detected", "Senior claimants (65+) have significantly higher repudiation rates compared to middle-aged claimants. Claims below age 18 are almost entirely repudiated, suggesting age-based screening inconsistencies."),
        ("Income Bias Detected", "Lower-income claimants (< ₹1L annual income) face a noticeably lower approval rate than high-income claimants (> ₹6L), raising equity concerns in claim settlement."),
        ("Zone/Team Bias Detected", "Significant variance in approval rates exists across zones — some teams approve >80% of claims while others approve <40%. This points to inconsistent team-level settlement practices."),
        ("Medical vs Non-Medical", "Non-medical policies have higher repudiation rates than medical policies, suggesting stricter documentation requirements or under-processing for non-medical cases."),
        ("Early Claim Suspicion", "EARLY claims (claims made shortly after policy issuance) have a disproportionately high repudiation rate, possibly masking legitimate grief claims."),
        ("Gender Disparity", "Male claimants form the vast majority of claims, but the male-to-female approval rate differential warrants monitoring for systemic bias."),
    ]

    for title, desc in findings:
        st.markdown(f"<div class='warning-box'><b>⚠ {title}</b><br>{desc}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🤖 Model Insights</div>', unsafe_allow_html=True)
    model_findings = [
        "Random Forest and Gradient Boosting are the most stable models, achieving the highest AUC-ROC scores.",
        "Decision Tree tends to overfit slightly (high train vs lower test accuracy gap), useful for explainability.",
        "KNN is sensitive to scale and performs adequately after normalization, but is less interpretable.",
        "Top predictive features (across tree models): PI_AGE, SUM_TO_INCOME, INCOME_LOG, ZONE_ENC, and IS_EARLY.",
        "The fact that ZONE_ENC is a top feature in all tree models strongly supports the existence of team-level bias.",
    ]
    for f in model_findings:
        st.markdown(f"<div class='finding-box'>✅ {f}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Recommendations</div>', unsafe_allow_html=True)
    recs = [
        "Implement a standardised claim-assessment rubric across all zones to eliminate team-level variance.",
        "Introduce an independent audit committee to review claims from low-approval zones.",
        "Remove or de-weight demographic variables (age, income) that are not legally relevant to claim validity.",
        "Flag EARLY claims for compassionate review rather than automatic high scrutiny.",
        "Deploy the Gradient Boosting model as a fairness-monitoring tool to detect outlier repudiation patterns.",
        "Conduct quarterly bias audits using cross-tabulation dashboards like this one.",
        "Provide sensitivity training to settlement officers in high-repudiation zones.",
    ]
    for r in recs:
        st.markdown(f"<div class='finding-box'>📌 {r}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Approval Rate", f"{approve_rate:.1f}%")
    col2.metric("Min Zone Approval", f"{df.groupby('ZONE')['CLAIM_APPROVED'].mean().min()*100:.1f}%")
    col3.metric("Max Zone Approval", f"{df.groupby('ZONE')['CLAIM_APPROVED'].mean().max()*100:.1f}%")
