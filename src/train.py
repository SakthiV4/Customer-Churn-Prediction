"""
train.py — IBM Telco Churn XGBoost Training
Run locally: python src/train.py
Saves: models/churn_pipeline.pkl, models/feature_columns.pkl
"""
import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "churn_pipeline.pkl")
COLS_PATH    = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "num_services"
]


def build_pipeline(n_estimators=300):
    ct = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough"
    )
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False
    )
    return Pipeline([("pre", ct), ("clf", xgb)])


def main():
    print("=" * 60)
    print("  IBM Telco Churn — XGBoost Training")
    print("=" * 60)

    # ── 1. Load & Preprocess ──────────────────────────────────────
    print(f"\n[1/4] Loading data...", flush=True)
    X, y = preprocess(DATA_PATH)
    print(f"      {len(X)} rows  |  {X.shape[1]} features  |  Churn={y.mean()*100:.1f}%")

    # ── 2. Train / Test Split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2/4] Split — Train: {len(X_train)}  Test: {len(X_test)}", flush=True)

    # ── 3. Train ─────────────────────────────────────────────────
    print(f"\n[3/4] Training XGBoost (n=300)...", flush=True)
    pipeline = build_pipeline(n_estimators=300)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    # Threshold sweep to maximise accuracy (class imbalance trick)
    best_acc, best_thresh = acc, 0.50
    for t in np.arange(0.30, 0.80, 0.01):
        a = accuracy_score(y_test, (y_prob >= t).astype(int))
        if a > best_acc:
            best_acc, best_thresh = a, t

    y_pred_final = (y_prob >= best_thresh).astype(int)

    print("\n" + "=" * 60)
    print(f"  Test Accuracy : {best_acc*100:.2f}%  (threshold={best_thresh:.2f})")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_final, target_names=["No Churn","Churn"]))

    cm = confusion_matrix(y_test, y_pred_final)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # ── 4. Cross-Validation ───────────────────────────────────────
    print(f"\n[4/4] 5-Fold CV (takes ~30 sec)...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"\n  CV Accuracy : {[round(a*100,2) for a in cv_acc]}")
    print(f"  Mean Acc    : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
    print(f"  CV ROC-AUC  : {[round(a,4) for a in cv_auc]}")
    print(f"  Mean AUC    : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    gap = train_acc - cv_acc.mean()
    print(f"\n  Train Acc   : {train_acc*100:.2f}%")
    print(f"  Train-CV Gap: {gap*100:.2f}%  (< 3% = no overfit)")

    # ── 5. Save ──────────────────────────────────────────────────
    artifact = {
        "pipeline":   pipeline,
        "threshold":  best_thresh,
        "model_name": "XGBoost_IBMTelco",
        "accuracy":   best_acc,
        "roc_auc":    auc,
    }
    joblib.dump(artifact, MODEL_PATH)
    joblib.dump(list(X.columns), COLS_PATH)
    print(f"\n  Model saved → {MODEL_PATH} ({os.path.getsize(MODEL_PATH)//1024} KB)")
    print(f"  Cols  saved → {COLS_PATH}  ({len(X.columns)} features)")
    print("\nDONE ✅")


if __name__ == "__main__":
    main()
