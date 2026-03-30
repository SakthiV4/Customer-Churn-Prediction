"""
preprocess.py — IBM Telco Customer Churn Dataset Preprocessing
7,043 rows | 21 raw features → 48 engineered features | Target: Churn (Yes=1, No=0)
"""
import pandas as pd
import numpy as np


def preprocess(filepath: str):
    """
    Load and preprocess the IBM Telco Customer Churn dataset.

    Returns
    -------
    X : pd.DataFrame  — Feature matrix (48 features)
    y : pd.Series     — Target (1=Churn, 0=No Churn)
    """
    df = pd.read_csv(filepath)
    print(f"[preprocess] Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 1. Drop customer ID ──────────────────────────────────────
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # ── 2. Fix TotalCharges (stored as string with spaces) ───────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[preprocess] After TotalCharges fix: {len(df)} rows")

    # ── 3. Encode target ─────────────────────────────────────────
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # ── 4. Feature Engineering ───────────────────────────────────
    # Tenure groups (customer loyalty buckets)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
    ).astype(str)

    # Count number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service",
                                                     "No phone service"]), axis=1
    )

    # Average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Month-to-month contract flag (highest churn risk)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # ── 5. Binary/Label encode low-cardinality categorical cols ─
    binary_map = {"Yes": 1, "No": 0,
                  "Male": 1, "Female": 0,
                  "No internet service": 0,
                  "No phone service": 0}

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "SeniorCitizen"
    ]
    # SeniorCitizen is already 0/1
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(binary_map).fillna(df[col]).astype(int)

    # ── 6. One-Hot Encode multi-class categoricals ───────────────
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod", "tenure_group"
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # ── 7. Split features / target ───────────────────────────────
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # Ensure bool → int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"[preprocess] Final feature matrix: {X.shape}")
    print(f"[preprocess] Churn distribution: {y.value_counts().to_dict()}")
    return X, y

# Triggered for CodeRabbit Code Review
