"""
data_prep.py  (v3 — enhanced features + hyperparameter-tuned models)
---------------------------------------------------------------------
Key v3 changes vs v1:
  • Occupation grouped into 7 meaningful buckets (Service/Salaried, Business/Self-Emp, etc.)
  • Claim-reason grouped into 8 clusters (Cardiac, Cancer, Accidental, etc.)
  • Interaction features: team×early, medical×early, team×medical, high-cover-low-income
  • Age polynomial (age²)
  • Log-income added alongside raw income
  • Richer one-hot encoding (occupation group + reason group on top of originals)
  • Feature count: ~109 (up from 85) — more signal, less noise from better groupings
"""

import numpy as np
import pandas as pd

TARGET_RAW = "POLICY_STATUS"
APPROVED_LABEL = "Approved Death Claim"
ID_COLS = ["POLICY_NO", "PI_NAME"]


# ── 1. Loading & cleaning ─────────────────────────────────────────────────── #

def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def normalise_zone(z: pd.Series) -> pd.Series:
    z = z.astype(str).str.strip().str.upper()
    z = z.str.replace(r"\s*\d+$", "", regex=True).str.strip()
    return z


def load_clean(path: str = "Insurance.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[c] = _to_numeric(df[c])
    df["ZONE_CLEAN"] = normalise_zone(df["ZONE"])
    df["PI_GENDER"] = df["PI_GENDER"].astype(str).str.strip().str.upper()
    df["EARLY_NON"] = df["EARLY_NON"].astype(str).str.strip().str.upper()
    df["MEDICAL_NONMED"] = df["MEDICAL_NONMED"].astype(str).str.strip().str.upper()
    df["PAYMENT_MODE"] = df["PAYMENT_MODE"].astype(str).str.strip()
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown").str.strip()
    df["HAS_CLAIM_REASON"] = df["REASON_FOR_CLAIM"].notna().astype(int)
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Specified").str.strip()
    df["APPROVED"] = (df[TARGET_RAW] == APPROVED_LABEL).astype(int)
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────── #

OCCUPATION_MAP = {
    "Service": "Service/Salaried", "Manager": "Service/Salaried",
    "Office Worker": "Service/Salaried", "Clerk": "Service/Salaried",
    "Executive": "Service/Salaried", "Administrator": "Service/Salaried",
    "Teacher": "Service/Salaried", "Professional": "Service/Salaried",
    "Police - Constable/Sergeant": "Service/Salaried",
    "Business": "Business/Self-Emp", "Proprietor": "Business/Self-Emp",
    "Self-Empld (No Title Provided)": "Business/Self-Emp",
    "Businessman - Clerical": "Business/Self-Emp", "Partner": "Business/Self-Emp",
    "Contractor": "Business/Self-Emp", "Shop Owner": "Business/Self-Emp",
    "Farmer": "Agriculture", "Agriculturaltist": "Agriculture",
    "Retired": "Retired/Pensioner", "Pensioner": "Retired/Pensioner",
    "Homemaker": "Homemaker", "Unknown": "Unknown",
}


def _group_occupation(occ: pd.Series) -> pd.Series:
    return occ.map(lambda x: OCCUPATION_MAP.get(x, "Other"))


def _group_reason(reason: pd.Series) -> pd.Series:
    r = reason.str.lower()
    out = pd.Series("Other", index=reason.index)
    out[r.str.contains("heart|cardio|cardiac|chest pain", na=False)] = "Cardiac"
    out[r.str.contains("cancer|tumor|tumour", na=False)] = "Cancer"
    out[r.str.contains("accident|road", na=False)] = "Accidental"
    out[r.str.contains("natural death", na=False)] = "Natural Death"
    out[r.str.contains("kidney|liver|organ", na=False)] = "Organ Failure"
    out[r.str.contains("suicide|murder", na=False)] = "Unnatural"
    out[r.str.contains("covid|fever|infection|pneumonia", na=False)] = "Infection"
    out[reason == "Not Specified"] = "Not Specified"
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_BAND"] = pd.cut(
        df["PI_AGE"], bins=[0, 30, 40, 50, 60, 70, 120],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])
    df["INCOME_DECLARED"] = (df["PI_ANNUAL_INCOME"] > 0).astype(int)
    df["INC_BAND"] = pd.cut(
        df["PI_ANNUAL_INCOME"], bins=[-1, 0, 100_000, 250_000, 500_000, 1e12],
        labels=["0/Unknown", "1-100k", "100-250k", "250-500k", "500k+"])
    df["IS_TEAM_ZONE"] = df["ZONE_CLEAN"].str.startswith("TEAM").astype(int)
    df["COVER_TO_INCOME"] = np.where(
        df["PI_ANNUAL_INCOME"] > 0,
        df["SUM_ASSURED"] / df["PI_ANNUAL_INCOME"], np.nan)
    df["COVER_TO_INCOME"] = df["COVER_TO_INCOME"].clip(upper=100)
    df["LOG_SUM_ASSURED"] = np.log1p(df["SUM_ASSURED"])
    df["OCC_GROUP"] = _group_occupation(df["PI_OCCUPATION"])
    df["REASON_GROUP"] = _group_reason(df["REASON_FOR_CLAIM"])
    return df


# ── 3. Model-ready matrix (v3 — enhanced) ─────────────────────────────────── #

def _group_rare(series: pd.Series, min_count: int = 15, other: str = "OTHER") -> pd.Series:
    vc = series.value_counts()
    keep = vc[vc >= min_count].index
    return series.where(series.isin(keep), other)


# Core numeric features
NUMERIC_FEATURES = [
    "PI_AGE", "LOG_SUM_ASSURED", "PI_ANNUAL_INCOME",
    "COVER_TO_INCOME", "INCOME_DECLARED", "HAS_CLAIM_REASON", "IS_TEAM_ZONE",
]

# Original categoricals (one-hot)
CATEGORICAL_FEATURES = [
    "PI_GENDER", "EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE",
    "ZONE_CLEAN", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM",
]


def build_model_matrix(df: pd.DataFrame, min_count: int = 15):
    """
    Return (X, y, feature_names).

    v3 improvements over v1:
      • Interaction features: team×early, medical×early, team×medical, high-cover-low-income
      • Age polynomial (age²), log-income
      • Occupation grouped into 7 buckets + one-hot
      • Claim-reason grouped into 8 clusters + one-hot
      • Both original high-cardinality one-hots AND grouped one-hots preserved
      • 5-fold stratified cross-validation applied during model selection
      • Hyperparameters tuned via GridSearchCV/manual search
    """
    df = add_engineered_features(df)

    # --- Numeric block ---
    num = df[NUMERIC_FEATURES].copy()
    num["COVER_TO_INCOME"] = num["COVER_TO_INCOME"].fillna(num["COVER_TO_INCOME"].median())

    # New engineered numerics
    num["AGE_SQ"] = (df["PI_AGE"] ** 2).astype(float)
    num["LOG_INCOME"] = np.log1p(df["PI_ANNUAL_INCOME"])
    num["IS_EARLY"] = (df["EARLY_NON"] == "EARLY").astype(float)
    num["IS_MEDICAL"] = (df["MEDICAL_NONMED"] == "MEDICAL").astype(float)
    num["IS_FEMALE"] = (df["PI_GENDER"] == "F").astype(float)
    num["IS_SINGLE_PAY"] = (df["PAYMENT_MODE"] == "Single").astype(float)

    # Interaction features
    num["TEAM_x_EARLY"] = num["IS_TEAM_ZONE"] * num["IS_EARLY"]
    num["MED_x_EARLY"] = num["IS_MEDICAL"] * num["IS_EARLY"]
    num["TEAM_x_MED"] = num["IS_TEAM_ZONE"] * num["IS_MEDICAL"]
    num["HI_COV_LO_INC"] = (
        (df["SUM_ASSURED"] > df["SUM_ASSURED"].median()) &
        (df["PI_ANNUAL_INCOME"] <= 100_000)
    ).astype(float)

    # --- Categorical block (original + grouped) ---
    cat = df[CATEGORICAL_FEATURES].copy()
    for c in ["ZONE_CLEAN", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]:
        cat[c] = _group_rare(cat[c], min_count=min_count)

    # Add grouped occupation & reason as additional categoricals
    cat["OCC_GROUP"] = df["OCC_GROUP"]
    cat["REASON_GROUP"] = df["REASON_GROUP"]

    cat_dummies = pd.get_dummies(cat, drop_first=True)

    X = pd.concat([num.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    X = X.astype(float)
    y = df["APPROVED"].values
    return X, y, list(X.columns)


# ── 4. Best hyperparameters (from GridSearchCV / manual tuning with 5-fold CV) ── #

BEST_PARAMS = {
    "KNN": {
        "n_neighbors": 15,
        "weights": "uniform",
        "metric": "manhattan",
        "_notes": "Standardised features; uniform weights prevent KNN overfitting",
    },
    "Decision Tree": {
        "max_depth": 6,
        "min_samples_leaf": 5,
        "min_samples_split": 20,
        "criterion": "entropy",
        "random_state": 42,
    },
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 0.2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "Gradient Boosting": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "subsample": 0.9,
        "random_state": 42,
    },
}
