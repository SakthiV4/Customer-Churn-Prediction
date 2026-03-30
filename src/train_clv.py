"""
train_clv.py — Customer Lifetime Value (CLV) Regression Model
==============================================================
This is a COMPLETELY ISOLATED training script.
It does NOT touch churn_pipeline.pkl or feature_columns.pkl.
It saves its own separate artifact: models/clv_pipeline.pkl

Target: Predicted Future Revenue = (Predicted Remaining Tenure) * MonthlyCharges
Algorithm: Gradient Boosting Regressor (sklearn) — no extra dependencies needed.

Run: python src/train_clv.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Safety: catch ALL errors so we never break the project ──────────────────
try:
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import mean_absolute_error, r2_score

except ImportError as e:
    print(f"\n❌ CLV Training SKIPPED — Missing package: {e}")
    print("   (This does NOT affect your main churn model.)\n")
    sys.exit(0)

# ── Paths — all isolated, nothing shared with churn pipeline ────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLV_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clv_pipeline.pkl")
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV. We work directly on raw data — no dependency on preprocess.py."""
    df = pd.read_csv(path)
    # Fix TotalCharges — it contains spaces for brand new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)
    return df


def engineer_clv_target(df: pd.DataFrame) -> pd.Series:
    """
    CLV Target = Future Revenue the customer will generate.
    We approximate this using:
        predicted_max_tenure = 72  (the dataset cap)
        remaining_months     = max(0, predicted_max_tenure - current_tenure)
        CLV                  = remaining_months * MonthlyCharges
    For customers who already churned, CLV is 0.
    """
    MAX_TENURE = 72
    already_churned = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    remaining = (MAX_TENURE - df["tenure"]).clip(lower=0)
    clv = remaining * df["MonthlyCharges"] * (1 - already_churned)
    return clv.astype(float)


def build_clv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and preprocess features for CLV prediction. Kept simple and safe."""
    features = df[[
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
        "Contract", "InternetService", "PaymentMethod",
        "Partner", "Dependents", "PaperlessBilling",
        "PhoneService", "OnlineSecurity", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]].copy()
    return features


def build_clv_pipeline() -> Pipeline:
    """Build an isolated sklearn pipeline for CLV regression."""
    numeric_cols  = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = [
        "Contract", "InternetService", "PaymentMethod",
        "Partner", "Dependents", "PaperlessBilling",
        "PhoneService", "OnlineSecurity", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    preprocessor = ColumnTransformer([
        ("num",  StandardScaler(), numeric_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])

    regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        loss="absolute_error"  # Robust to outliers (was 'huber' in older sklearn)
    )

    return Pipeline([
        ("pre", preprocessor),
        ("reg", regressor)
    ])


def main():
    print("=" * 60)
    print("  CLV Regression Model Training (ISOLATED)")
    print("  ⚠️  Churn model will NOT be affected.")
    print("=" * 60)

    # ── 1. Load raw data ────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Data not found: {DATA_PATH}")
        print("   CLV training skipped. Churn model is safe.\n")
        sys.exit(0)

    print(f"\n[1/4] Loading raw data from {DATA_PATH}...")
    df = load_raw_data(DATA_PATH)
    print(f"      {len(df)} rows loaded.")

    # ── 2. Build features and target ────────────────────────────────
    print("\n[2/4] Engineering CLV target...")
    X = build_clv_features(df)
    y = engineer_clv_target(df)
    print(f"      Features: {X.shape[1]}  |  CLV range: ${y.min():.0f} – ${y.max():.0f}")
    print(f"      Average predicted CLV: ${y.mean():.2f}")

    # ── 3. Train/Test Split & Train ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[3/4] Training GradientBoosting Regressor...")
    pipeline = build_clv_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  Mean Absolute Error : ${mae:.2f}")
    print(f"  R² Score            : {r2:.4f}  (1.0 = perfect)")
    print("=" * 60)

    # ── 4. Cross-Validation ─────────────────────────────────────────
    print("\n[4/4] Running 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=-1)
    cv_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)

    print(f"\n  CV R² scores : {[round(s, 4) for s in cv_r2]}")
    print(f"  Mean CV R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Mean CV MAE  : ${abs(cv_mae.mean()):.2f}")

    # ── 5. Save — completely isolated artifact ───────────────────────
    clv_artifact = {
        "pipeline":   pipeline,
        "model_name": "CLV_GradientBoost",
        "mae":        mae,
        "r2":         r2,
        "avg_clv":    float(y.mean()),
        "max_clv":    float(y.max()),
    }
    joblib.dump(clv_artifact, CLV_MODEL_PATH)
    size_kb = os.path.getsize(CLV_MODEL_PATH) // 1024
    print(f"\n  ✅ CLV model saved → {CLV_MODEL_PATH} ({size_kb} KB)")
    print("  ✅ Churn model untouched. Both models can run side-by-side.")
    print("\nDONE ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CLV Training encountered an error: {e}")
        print("   (This does NOT affect your main churn model.)")
        print("   Please check the error above and try again.\n")
        sys.exit(0)
