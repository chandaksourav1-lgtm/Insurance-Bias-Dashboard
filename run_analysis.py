"""
run_analysis.py
---------------
Runs the complete analysis end-to-end and writes every required artefact
(cross-tabs, diagnostic charts, model metrics, ROC curves, confusion
matrices, findings) to the ./outputs folder. This is the "run it here"
counterpart to the Streamlit dashboard and uses the exact same data_prep.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RND = 42


# --------------------------------------------------------------------------- #
def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def rate_table(df, col):
    t = (df.groupby(col, observed=True)["APPROVED"]
           .agg(approval_rate="mean", claims="count")
           .sort_values("approval_rate"))
    t["approval_rate"] = (t["approval_rate"] * 100).round(1)
    return t


# --------------------------------------------------------------------------- #
def descriptive_and_diagnostic(df):
    print("Descriptive + diagnostic analysis ...")
    seg_cols = ["EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE", "PI_GENDER",
                "AGE_BAND", "INC_BAND", "IS_TEAM_ZONE"]

    # ---- cross-tab heatmap of approval rate by segment ----
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for c in ["EARLY_NON", "MEDICAL_NONMED", "PI_GENDER", "IS_TEAM_ZONE"]:
        for lvl, sub in df.groupby(c, observed=True):
            rows.append((f"{c}={lvl}", sub["APPROVED"].mean() * 100, len(sub)))
    seg = pd.DataFrame(rows, columns=["segment", "rate", "n"]).set_index("segment")
    sns.heatmap(seg[["rate"]], annot=True, fmt=".1f", cmap="RdYlGn",
                cbar_kws={"label": "Approval rate %"}, ax=ax, vmin=40, vmax=100)
    ax.set_title("Approval rate (%) by binary segment")
    ax.set_xlabel("")
    save(fig, "01_segment_heatmap.png")

    # ---- zone / team approval bar ----
    z = rate_table(df, "ZONE_CLEAN")
    z = z[z["claims"] >= 15]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2ca02c" if df[df.ZONE_CLEAN == idx]["IS_TEAM_ZONE"].iloc[0] == 1
              else "#1f77b4" for idx in z.index]
    ax.barh(z.index, z["approval_rate"], color=colors)
    ax.axvline(df["APPROVED"].mean() * 100, color="red", ls="--",
               label=f"Overall {df['APPROVED'].mean()*100:.1f}%")
    ax.set_xlabel("Approval rate %")
    ax.set_title("Approval rate by zone/channel  (green = TEAM channel)")
    ax.legend()
    save(fig, "02_zone_approval.png")

    # ---- age & income diagnostic ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    a = rate_table(df, "AGE_BAND").sort_index()
    axes[0].bar(a.index.astype(str), a["approval_rate"], color="#4c72b0")
    axes[0].axhline(df["APPROVED"].mean() * 100, color="red", ls="--")
    axes[0].set_title("Approval rate by age band")
    axes[0].set_ylabel("Approval rate %")
    i = rate_table(df, "INC_BAND").sort_index()
    axes[1].bar(i.index.astype(str), i["approval_rate"], color="#dd8452")
    axes[1].axhline(df["APPROVED"].mean() * 100, color="red", ls="--")
    axes[1].set_title("Approval rate by income band")
    save(fig, "03_age_income_diagnostic.png")

    # ---- chi-square style contribution: team x early ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ct = pd.crosstab(df["IS_TEAM_ZONE"].map({0: "Non-TEAM", 1: "TEAM"}),
                     df["POLICY_STATUS"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Outcome mix: TEAM vs Non-TEAM channels")
    ax.set_ylabel("% of claims")
    ax.set_xlabel("")
    ax.legend(title="", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=0)
    save(fig, "04_team_outcome_mix.png")

    return seg, z


# --------------------------------------------------------------------------- #
def train_models(df):
    print("Feature engineering + model training ...")
    X, y, names = dp.build_model_matrix(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RND, stratify=y)

    models = {
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=15))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                random_state=RND),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=None,
                                                min_samples_leaf=5, n_jobs=-1,
                                                random_state=RND),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RND),
    }

    results, roc_data, conf = {}, {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_tr, p_te = m.predict(X_tr), m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        results[name] = {
            "train_acc": accuracy_score(y_tr, p_tr),
            "test_acc": accuracy_score(y_te, p_te),
            "precision": precision_score(y_te, p_te),
            "recall": recall_score(y_te, p_te),
            "f1": f1_score(y_te, p_te),
            "roc_auc": roc_auc_score(y_te, proba),
        }
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_data[name] = (fpr, tpr, results[name]["roc_auc"])
        conf[name] = confusion_matrix(y_te, p_te)
        print(f"  {name:18s} testAcc={results[name]['test_acc']:.3f} "
              f"AUC={results[name]['roc_auc']:.3f}")

    metrics = pd.DataFrame(results).T
    return metrics, roc_data, conf, models, X, names, (X_tr, X_te, y_tr, y_te)


def plot_model_outputs(metrics, roc_data, conf, models, X):
    # ---- metric comparison bars ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics[["train_acc", "test_acc"]].plot(kind="bar", ax=axes[0],
                                            color=["#9ecae1", "#3182bd"])
    axes[0].set_title("Training vs Testing accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    metrics[["precision", "recall", "f1"]].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Precision / Recall / F1 (test set)")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)
    save(fig, "05_metric_comparison.png")

    # ---- ROC curves ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — all models")
    ax.legend(loc="lower right")
    save(fig, "06_roc_curves.png")

    # ---- confusion matrices ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, cm) in zip(axes.ravel(), conf.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Repudiate", "Approve"],
                    yticklabels=["Repudiate", "Approve"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices (test set)", y=1.01, fontsize=13)
    save(fig, "07_confusion_matrices.png")

    # ---- feature importance (Random Forest) ----
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp[::-1].plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title("Top 15 drivers of claim approval (Random Forest importance)")
    save(fig, "08_feature_importance.png")
    return imp


# --------------------------------------------------------------------------- #
def main():
    df = dp.add_engineered_features(dp.load_clean("Insurance.csv"))
    seg, z = descriptive_and_diagnostic(df)
    metrics, roc_data, conf, models, X, names, splits = train_models(df)
    imp = plot_model_outputs(metrics, roc_data, conf, models, X)

    metrics.round(4).to_csv(OUT / "model_metrics.csv")
    summary = {
        "overall_approval_rate": round(df["APPROVED"].mean() * 100, 1),
        "team_approval_rate": round(df[df.IS_TEAM_ZONE == 1]["APPROVED"].mean() * 100, 1),
        "nonteam_approval_rate": round(df[df.IS_TEAM_ZONE == 0]["APPROVED"].mean() * 100, 1),
        "best_model_auc": metrics["roc_auc"].idxmax(),
        "top_features": imp.head(8).round(4).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print("\nAll outputs written to ./outputs")


if __name__ == "__main__":
    main()
