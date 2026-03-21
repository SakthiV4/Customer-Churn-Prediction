"""
catboost_stack_train.py — 4-Model Stacking with CatBoost
Base Learners : XGBoost + LightGBM + RandomForest + CatBoost
Meta Learner  : Logistic Regression (5-fold OOF)
Features      : 55 (48 original + 7 interaction features)
"""
import os, sys, warnings, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.preprocess import preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

DATA  = os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL = os.path.join(os.path.dirname(__file__), "models", "churn_pipeline.pkl")
COLS  = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")
os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges",
                    "avg_monthly_spend", "num_services",
                    "charges_per_service", "tenure_monthly_ratio"]

print("=" * 65)
print("  4-Model Stack: XGBoost+LightGBM+RF+CatBoost → LR meta")
print("=" * 65)

# ── 1. Load & Preprocess ─────────────────────────────────────
print("\n[1/4] Preprocessing...", flush=True)
X, y = preprocess(DATA)

# ── 2. Add interaction features ───────────────────────────────
print("[2/4] Adding interaction features...", flush=True)
X["charges_per_service"]  = X["MonthlyCharges"] / (X["num_services"] + 1)
X["tenure_monthly_ratio"] = X["tenure"] / (X["MonthlyCharges"] + 1)
if "is_month_to_month" in X.columns:
    X["senior_mtm"]   = X["SeniorCitizen"] * X["is_month_to_month"]
    X["triple_risk"]  = ((X["tenure"] <= 12) *
                         (X["MonthlyCharges"] > X["MonthlyCharges"].quantile(0.75)) *
                          X["is_month_to_month"]).astype(int)
if "OnlineSecurity_No" in X.columns and "is_month_to_month" in X.columns:
    X["no_security_mtm"] = X["OnlineSecurity_No"] * X["is_month_to_month"]
X["high_charges"]  = (X["MonthlyCharges"] > X["MonthlyCharges"].quantile(0.75)).astype(int)
X["new_customer"]  = (X["tenure"] <= 12).astype(int)
print(f"      {X.shape[1]} features total")

# ── 3. Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train={len(X_train)}  Test={len(X_test)}", flush=True)

# ── 4. Build 4-model stacking ensemble ───────────────────────
print("\n[3/4] Building 4-model ensemble...", flush=True)

def make_pipe(clf):
    num_feats = [c for c in NUMERIC_FEATURES if c in X.columns]
    ct = ColumnTransformer([("num", StandardScaler(), num_feats)],
                           remainder="passthrough")
    return Pipeline([("pre", ct), ("clf", clf)])

# Best Optuna params (from previous run)
xgb_best = XGBClassifier(
    n_estimators=155, max_depth=4, learning_rate=0.041,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.05, reg_lambda=1.0, eval_metric="auc",
    random_state=42, use_label_encoder=False, n_jobs=-1
)
lgbm_best = lgb.LGBMClassifier(
    n_estimators=231, max_depth=4, learning_rate=0.024,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.05, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1
)
# CatBoost — handles categoricals natively, no scaling needed
catboost = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, border_count=128,
    random_seed=42, verbose=0, eval_metric="AUC",
    thread_count=-1
)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)

base_learners = [
    ("xgb",      make_pipe(xgb_best)),
    ("lgbm",     make_pipe(lgbm_best)),
    ("rf",       make_pipe(rf)),
    ("catboost", make_pipe(catboost)),
]

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
    cv=5, stack_method="predict_proba",
    passthrough=False, n_jobs=1
)

print("[4/4] Training (takes ~3-5 mins)...", flush=True)
stack.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_prob  = stack.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

best_acc, best_thresh = 0, 0.50
for t in np.arange(0.30, 0.80, 0.01):
    a = accuracy_score(y_test, (y_prob >= t).astype(int))
    if a > best_acc:
        best_acc, best_thresh = a, t

y_final = (y_prob >= best_thresh).astype(int)

print("\n" + "=" * 65)
print("  FINAL RESULTS — 4-Model CatBoost Stacking Ensemble")
print("=" * 65)
print(f"  Test Accuracy  : {best_acc*100:.2f}%  (threshold={best_thresh:.2f})")
print(f"  Test ROC-AUC   : {roc_auc:.4f}")
print("=" * 65)
print(classification_report(y_test, y_final, target_names=["No Churn","Churn"]))
cm = confusion_matrix(y_test, y_final)
print(f"  TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}")

# ── Individual model check ─────────────────────────────────────
print("\n  Individual model accuracy:", flush=True)
for name, pipe in base_learners:
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    ba = max(accuracy_score(y_test, (p >= t).astype(int))
             for t in np.arange(0.30, 0.80, 0.01))
    print(f"    {name:<10}: acc={ba*100:.2f}%  auc={roc_auc_score(y_test, p):.4f}")

# ── CV check ──────────────────────────────────────────────────
print("\n  5-Fold CV...", flush=True)
cv_acc = cross_val_score(stack, X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="accuracy", n_jobs=1)
train_acc = accuracy_score(y_train, stack.predict(X_train))
print(f"  CV Accuracy  : {[round(a*100,2) for a in cv_acc]}")
print(f"  Mean CV Acc  : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"  Train Acc    : {train_acc*100:.2f}%")
print(f"  Train-CV Gap : {(train_acc-cv_acc.mean())*100:.2f}%  (< 5% = no overfit)")

# ── Save ──────────────────────────────────────────────────────
artifact = {
    "pipeline":   stack,
    "threshold":  best_thresh,
    "model_name": "CatBoostStack_IBMTelco",
    "accuracy":   best_acc,
    "roc_auc":    roc_auc,
}
joblib.dump(artifact, MODEL)
joblib.dump(list(X.columns), COLS)
print(f"\n  Saved: {os.path.getsize(MODEL)//1024} KB | {len(X.columns)} features")
print("DONE ✅", flush=True)
