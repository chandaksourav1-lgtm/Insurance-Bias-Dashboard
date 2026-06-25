"""
data_prep.py
------------
Shared data-loading, cleaning and feature-engineering utilities for the
Insurance Claim-Settlement Bias analysis. Imported by both `run_analysis.py`
(static run) and `app.py` (Streamlit dashboard) so the two never drift apart.
"""

import numpy as np
import pandas as pd

TARGET_RAW = "POLICY_STATUS"
APPROVED_LABEL = "Approved Death Claim"

# Columns that are pure identifiers / leak-free-but-useless for modelling
ID_COLS = ["POLICY_NO", "PI_NAME"]


# --------------------------------------------------------------------------- #
# 1. Loading & cleaning
# --------------------------------------------------------------------------- #
def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip thousands separators / quotes and coerce to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def normalise_zone(z: pd.Series) -> pd.Series:
    """Merge casing duplicates (South/SOUTH, West/WEST, East/EAST n ...)."""
    z = z.astype(str).str.strip().str.upper()
    # collapse trailing region numbers e.g. "EAST 1" / "EAST 2" -> "EAST"
    z = z.str.replace(r"\s*\d+$", "", regex=True).str.strip()
    return z


def load_clean(path: str = "Insurance.csv") -> pd.DataFrame:
    """Read the raw CSV and return a cleaned dataframe (no encoding yet)."""
    df = pd.read_csv(path)

    # numeric money columns
    for c in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[c] = _to_numeric(df[c])

    # categorical tidy-up
    df["ZONE_CLEAN"] = normalise_zone(df["ZONE"])
    df["PI_GENDER"] = df["PI_GENDER"].astype(str).str.strip().str.upper()
    df["EARLY_NON"] = df["EARLY_NON"].astype(str).str.strip().str.upper()
    df["MEDICAL_NONMED"] = df["MEDICAL_NONMED"].astype(str).str.strip().str.upper()
    df["PAYMENT_MODE"] = df["PAYMENT_MODE"].astype(str).str.strip()

    # missing handling
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown").str.strip()
    df["HAS_CLAIM_REASON"] = df["REASON_FOR_CLAIM"].notna().astype(int)
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Specified").str.strip()

    # binary target: 1 = approved, 0 = repudiated
    df["APPROVED"] = (df[TARGET_RAW] == APPROVED_LABEL).astype(int)

    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analytical features (bands, flags, ratios)."""
    df = df.copy()

    df["AGE_BAND"] = pd.cut(
        df["PI_AGE"],
        bins=[0, 30, 40, 50, 60, 70, 120],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"],
    )

    # income is heavily zero-coded -> treat 0 as "not declared"
    df["INCOME_DECLARED"] = (df["PI_ANNUAL_INCOME"] > 0).astype(int)
    df["INC_BAND"] = pd.cut(
        df["PI_ANNUAL_INCOME"],
        bins=[-1, 0, 100_000, 250_000, 500_000, 1e12],
        labels=["0/Unknown", "1-100k", "100-250k", "250-500k", "500k+"],
    )

    # the segment under investigation: "TEAM ..." channels
    df["IS_TEAM_ZONE"] = df["ZONE_CLEAN"].str.startswith("TEAM").astype(int)

    # ratio of cover to declared income (capped); guards divide-by-zero
    df["COVER_TO_INCOME"] = np.where(
        df["PI_ANNUAL_INCOME"] > 0,
        df["SUM_ASSURED"] / df["PI_ANNUAL_INCOME"],
        np.nan,
    )
    df["COVER_TO_INCOME"] = df["COVER_TO_INCOME"].clip(upper=100)

    # log of sum assured (skew correction for distance-based models like KNN)
    df["LOG_SUM_ASSURED"] = np.log1p(df["SUM_ASSURED"])

    return df


def _group_rare(series: pd.Series, min_count: int = 15, other: str = "OTHER") -> pd.Series:
    """Collapse infrequent categories so one-hot encoding stays compact."""
    vc = series.value_counts()
    keep = vc[vc >= min_count].index
    return series.where(series.isin(keep), other)


# Features used for the predictive models
NUMERIC_FEATURES = [
    "PI_AGE",
    "LOG_SUM_ASSURED",
    "PI_ANNUAL_INCOME",
    "COVER_TO_INCOME",
    "INCOME_DECLARED",
    "HAS_CLAIM_REASON",
    "IS_TEAM_ZONE",
]
CATEGORICAL_FEATURES = [
    "PI_GENDER",
    "EARLY_NON",
    "MEDICAL_NONMED",
    "PAYMENT_MODE",
    "ZONE_CLEAN",
    "PI_STATE",
    "PI_OCCUPATION",
    "REASON_FOR_CLAIM",
]


def build_model_matrix(df: pd.DataFrame, min_count: int = 15):
    """
    Return (X, y, feature_names) ready for sklearn.
    High-cardinality categoricals have rare levels grouped, then everything
    is one-hot encoded. COVER_TO_INCOME NaNs are median-imputed.
    """
    df = add_engineered_features(df)

    cat = df[CATEGORICAL_FEATURES].copy()
    for c in ["ZONE_CLEAN", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]:
        cat[c] = _group_rare(cat[c], min_count=min_count)

    num = df[NUMERIC_FEATURES].copy()
    num["COVER_TO_INCOME"] = num["COVER_TO_INCOME"].fillna(num["COVER_TO_INCOME"].median())

    cat_dummies = pd.get_dummies(cat, drop_first=True)
    X = pd.concat([num.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    X = X.astype(float)
    y = df["APPROVED"].values
    return X, y, list(X.columns)
