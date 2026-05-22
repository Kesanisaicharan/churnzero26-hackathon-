"""
preprocess.py
=============
ChurnZero 26 — Banking Customer Churn Prediction
Feature Engineering & Preprocessing Pipeline

Team:
  - Kesani Sree Sai Charan Teja
  - Nagi Reddy SaiBaba
  - Jagriti Sharma
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    """Load train and test CSVs."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"Train shape : {train.shape}")
    print(f"Test  shape : {test.shape}")
    print(f"Churn rate  : {train['Exited'].mean()*100:.1f}%")
    return train, test


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Ratio features ---
    df["balance_salary_ratio"]  = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["products_per_tenure"]   = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["clv_estimate"]          = df["Balance"] * df["Tenure"]
    df["balance_per_product"]   = df["Balance"] / (df["NumOfProducts"] + 1)

    # --- Binary flags ---
    df["balance_zero_flag"]     = (df["Balance"] == 0).astype(int)
    df["high_balance_flag"]     = (df["Balance"] > df["Balance"].quantile(0.9)).astype(int)
    df["senior_flag"]           = (df["Age"] >= 55).astype(int)
    df["long_tenure_flag"]      = (df["Tenure"] >= 8).astype(int)
    df["multi_product_flag"]    = (df["NumOfProducts"] >= 3).astype(int)
    df["low_credit_flag"]       = (df["CreditScore"] < 500).astype(int)

    # --- Age segmentation (ordinal) ---
    df["age_segment"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    # --- Engagement score (composite) ---
    df["engagement_score"] = (
        df["IsActiveMember"] * 3 +
        df["NumOfProducts"]  * 2 +
        (df["Tenure"] / df["Tenure"].max()) * 2 +
        df["HasCrCard"]      * 1
    )

    # --- Interaction terms ---
    df["germany_inactive"]  = ((df["Geography"] == "Germany") & (df["IsActiveMember"] == 0)).astype(int)
    df["senior_inactive"]   = ((df["Age"] >= 55) & (df["IsActiveMember"] == 0)).astype(int)
    df["high_bal_inactive"] = (df["high_balance_flag"] == 1) & (df["IsActiveMember"] == 0)
    df["high_bal_inactive"] = df["high_bal_inactive"].astype(int)

    # --- Log transforms (handle skew) ---
    df["log_balance"] = np.log1p(df["Balance"])
    df["log_salary"]  = np.log1p(df["EstimatedSalary"])
    df["log_clv"]     = np.log1p(df["clv_estimate"])

    # --- Age squared (non-linearity) ---
    df["age_squared"] = df["Age"] ** 2

    return df


# ─────────────────────────────────────────────
# 3. ENCODING
# ─────────────────────────────────────────────
def encode_features(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode Geography & Gender."""
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    combined = pd.get_dummies(combined, columns=["Geography", "Gender"], drop_first=False)

    # Restore split
    n_train = len(train)
    train_enc = combined.iloc[:n_train].copy()
    test_enc  = combined.iloc[n_train:].copy()

    return train_enc, test_enc


# ─────────────────────────────────────────────
# 4. DROP IRRELEVANT COLUMNS
# ─────────────────────────────────────────────
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

FEATURE_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "balance_salary_ratio", "products_per_tenure", "clv_estimate",
    "balance_per_product", "balance_zero_flag", "high_balance_flag",
    "senior_flag", "long_tenure_flag", "multi_product_flag",
    "low_credit_flag", "age_segment", "engagement_score",
    "germany_inactive", "senior_inactive", "high_bal_inactive",
    "log_balance", "log_salary", "log_clv", "age_squared",
    # One-hot encoded cols added dynamically below
]

GEO_GENDER_COLS = [
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male"
]


def prepare_matrices(train: pd.DataFrame, test: pd.DataFrame, target: str = "Exited"):
    """Return X_train, y_train, X_test, feature_names."""

    train, test = encode_features(train, test)

    all_features = FEATURE_COLS + [c for c in GEO_GENDER_COLS if c in train.columns]

    # Drop columns not needed
    for col in DROP_COLS:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    X_train = train[all_features]
    y_train = train[target]
    X_test  = test[[c for c in all_features if c in test.columns]]

    return X_train, y_train, X_test, all_features


# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────
# MAIN — run standalone to verify pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <train.csv> <test.csv>")
        sys.exit(1)

    train_df, test_df = load_data(sys.argv[1], sys.argv[2])

    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    X_train, y_train, X_test, feats = prepare_matrices(train_df, test_df)
    X_train_sc, X_test_sc, scaler   = scale_features(X_train, X_test)

    print(f"\n✅ Feature matrix ready")
    print(f"   X_train : {X_train_sc.shape}")
    print(f"   X_test  : {X_test_sc.shape}")
    print(f"   Features: {feats}")
