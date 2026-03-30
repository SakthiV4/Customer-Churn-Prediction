import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List

# Load .env file when running locally (ignored on Vercel, uses env vars directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on Vercel

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
COLS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
CLV_MODEL_PATH = os.path.join(BASE_DIR, "models", "clv_pipeline.pkl")

# ── Global State ──────────────────────────────────────────────────────
app = FastAPI(
    title="RetentionLens AI - Action Platform",
    description="Advanced API with Churn Prediction, Feature Impact, CSV Batching, and AI Strategy Generation.",
    version="2.0.0"
)

# Allow CORS for local dev & Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend static files
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

pipeline = None
clv_pipeline = None
best_thresh = 0.47  

# ── Load Model on Startup ─────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global pipeline, best_thresh, clv_pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    try:
        artifact = joblib.load(MODEL_PATH)
        pipeline = artifact["pipeline"]
        best_thresh = artifact.get("threshold", 0.47)
        print(f"✅ Model loaded successfully (Threshold={best_thresh:.2f})")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
    
    # Load CLV model separately — safe, isolated, optional
    try:
        if os.path.exists(CLV_MODEL_PATH):
            clv_artifact = joblib.load(CLV_MODEL_PATH)
            clv_pipeline = clv_artifact["pipeline"]
            print(f"✅ CLV model loaded (avg CLV=${clv_artifact.get('avg_clv', 0):.0f})")
        else:
            print("ℹ️  CLV model not found — CLV feature will be skipped.")
    except Exception as e:
        print(f"⚠️  CLV model load failed (non-critical): {str(e)}")

# ── Pydantic Request Schema ───────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str = Field(..., description="Male or Female", example="Female")
    SeniorCitizen: int = Field(..., description="1 if senior citizen, 0 otherwise", example=0)
    Partner: str = Field(..., example="No")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=89.50)
    TotalCharges: float = Field(..., example=1074.0)


class EmailRequest(BaseModel):
    recipient_email: str = Field(..., example="customer@example.com")
    subject: str = Field(..., example="We value your partnership!")
    body: str = Field(..., example="Hi, we have a special offer for you...")


# ── Feature Engineering Logic ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        include_lowest=True,
        labels=["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
    ).astype(str)

    # 2. Count number of services
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]), axis=1
    )

    # 3. Average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 4. Month-to-month flag
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 5. Binary Encodings
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "No internet service": 0, "No phone service": 0}
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(df[col]).astype(int)

    # 6. One-Hot Encodings
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod", "tenure_group"
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # 7. Interaction Features
    df["charges_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    df["tenure_monthly_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1)
    if "is_month_to_month" in df.columns:
        df["senior_mtm"] = df["SeniorCitizen"] * df["is_month_to_month"]
        q75 = 89.85
        df["triple_risk"] = ((df["tenure"] <= 12) * (df["MonthlyCharges"] > q75) * df["is_month_to_month"]).astype(int)
    if "OnlineSecurity_No" in df.columns and "is_month_to_month" in df.columns:
        df["no_security_mtm"] = df["OnlineSecurity_No"] * df["is_month_to_month"]
    df["high_charges"] = (df["MonthlyCharges"] > 89.85).astype(int)
    df["new_customer"] = (df["tenure"] <= 12).astype(int)

    # 8. Align columns
    try:
        expected_cols = joblib.load(COLS_PATH)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    except Exception as e:
        print(f"Warning: feature_columns.pkl not found. {e}")
        
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

# ── Feature Impact Analyzer (Fast Perturbation) ───────────────────────
def calculate_feature_impacts(original_df: pd.DataFrame, base_prob: float) -> list:
    """Calculates how much key raw features contributed to the churn score."""
    impacts = []
    
    # Define perturbation scenarios for high-risk inputs
    tests = [
        {"feature": "Contract", "value": "Two year", "label": "Contract Type", "reason": "Month-to-month contracts lack lock-in and make it easy to cancel."},
        {"feature": "InternetService", "value": "DSL", "label": "Internet Speed", "reason": "Expensive Fiber optic plans increase price sensitivity."},
        {"feature": "MonthlyCharges", "value": original_df["MonthlyCharges"].iloc[0] * 0.7, "label": "Monthly Bill", "reason": "High monthly charges cause customers to seek cheaper alternatives."},
        {"feature": "tenure", "value": 72, "label": "Loyalty / Tenure", "reason": "Brand new customers haven't built brand loyalty yet."},
        {"feature": "TechSupport", "value": "Yes", "label": "Tech Support", "reason": "Lack of technical support causes unresolved customer frustration."}
    ]
    
    for t in tests:
        orig_val = original_df[t["feature"]].iloc[0]
        # Only test if it's currently a "risk" factor
        if orig_val != t["value"]:
            test_df = original_df.copy()
            test_df.at[0, t["feature"]] = t["value"]
            
            # Re-engineer and predict
            X_test = engineer_features(test_df)
            new_prob = float(pipeline.predict_proba(X_test)[0][1])
            
            # The impact is how much the probability DROPPED when we fixed the risk
            impact = base_prob - new_prob 
            if impact > 0.01: # Cap at meaningful impact
                impacts.append({
                    "feature": t["label"],
                    "value": f"{orig_val}",
                    "impact_percentage": round(impact * 100, 1),
                    "type": "negative", # It drove churn UP
                    "reason": t["reason"]
                })
                
    # Sort by highest impact first
    impacts = sorted(impacts, key=lambda x: x["impact_percentage"], reverse=True)[:3]
    return impacts

# ── LLM Strategy Generator (Rule-based Dynamic Scripting) ─────────────
def generate_retention_strategy(prob: float, impacts: list, raw_data: dict) -> str:
    """Dynamically writes a retention script mimicking an LLM using the Impact Analysis."""
    if prob < best_thresh:
        return "This customer is loyal and safe. No action needed right now!"
    
    script = f"🔥 AI Insight: "
    
    if not impacts:
        return script + f"They are at risk. Reach out and see how we can help them."
        
    top_reason = impacts[0]["feature"]
    
    # Get conversational context
    is_new = "brand new " if raw_data["tenure"] <= 12 else ""
    is_expensive = "expensive " if raw_data["MonthlyCharges"] > 70 else ""
    contract_type = raw_data["Contract"].lower()
    internet_type = raw_data["InternetService"]
    
    if "Contract" in top_reason:
        return script + f'"This is a {is_new}customer paying for {is_expensive}{internet_type} on a {contract_type} contract. They are going to cancel soon. You need to lock them into a 1-year contract right now to save them."'
    elif "Bill" in top_reason:
        return script + f'"This customer is frustrated by their {is_expensive}monthly bill. They are looking for a cheaper option. You need to immediately apply a retention credit to lower their bill and save the account."'
    elif "Tenure" in top_reason or "Loyalty" in top_reason:
        return script + f'"This is a brand new customer who hasn\'t formed a habit yet. They are experiencing friction. You need to assign a technical specialist to hold their hand and make sure everything is working."'
    elif "Support" in top_reason:
        return script + f'"This customer doesn\'t have any technical support and is likely struggling. You need to call them and offer a free 3-month trial of Premium Tech Support and run a diagnostic on their equipment."'
    else:
        return script + f'"This customer is at high risk. Highlight how much they\'ve spent with us to date (${raw_data["TotalCharges"]}) and offer a loyalty discount to convince them to stay."'

# ── API Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def serve_index():
    return RedirectResponse(url="/static/index.html")

@app.get("/batch")
def serve_batch_ui():
    return RedirectResponse(url="/static/batch.html")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "threshold": best_thresh
    }

@app.post("/api/predict")
def predict_churn(data: CustomerData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        # 1. Base Prediction
        X = engineer_features(df)
        churn_prob = float(pipeline.predict_proba(X)[0][1])
        will_churn = bool(churn_prob >= best_thresh)

        # 2. Advanced: Feature Impacts
        impacts = calculate_feature_impacts(df, churn_prob)
        
        # 3. Advanced: Strategy Generator
        strategy = generate_retention_strategy(churn_prob, impacts, input_data)

        # 4. CLV Prediction (isolated, non-breaking)
        clv_data = None
        try:
            if clv_pipeline is not None:
                clv_input = df[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
                                'Contract','InternetService','PaymentMethod','Partner',
                                'Dependents','PaperlessBilling','PhoneService','OnlineSecurity',
                                'TechSupport','StreamingTV','StreamingMovies']].copy()
                predicted_clv = float(clv_pipeline.predict(clv_input)[0])
                predicted_clv = max(0, round(predicted_clv, 2))
                # Priority tier based on CLV
                if predicted_clv >= 3000:
                    clv_tier = "VIP"
                    clv_color = "#a855f7"
                elif predicted_clv >= 1500:
                    clv_tier = "High Value"
                    clv_color = "#3b82f6"
                elif predicted_clv >= 500:
                    clv_tier = "Standard"
                    clv_color = "#22c55e"
                else:
                    clv_tier = "Low Value"
                    clv_color = "#6b7280"
                clv_data = {
                    "predicted_clv": predicted_clv,
                    "tier": clv_tier,
                    "color": clv_color,
                    "monthly_charges": input_data["MonthlyCharges"]
                }
        except Exception as clv_err:
            print(f"⚠️ CLV prediction skipped: {clv_err}")

        # UI Payload
        if churn_prob >= 0.70:
            risk_level = "Critical Risk"
            color = "#ef4444"
        elif churn_prob >= best_thresh:
            risk_level = "High Risk"
            color = "#f97316"
        elif churn_prob >= best_thresh - 0.15:
            risk_level = "Medium Risk"
            color = "#eab308"
        else:
            risk_level = "Low Risk"
            color = "#22c55e"

        return {
            "prediction": "Churn" if will_churn else "Retain",
            "probability": round(churn_prob, 4),
            "threshold": round(float(best_thresh), 4),
            "risk_level": risk_level,
            "colorCode": color,
            "message": "Customer is highly likely to churn." if will_churn else "Customer is securely retained.",
            "impacts": impacts,
            "retention_strategy": strategy,
            "clv": clv_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/batch")
async def predict_churn_batch(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        df = pd.read_csv(file.file)
        
        # Keep original indices/IDs if present
        customer_ids = df.get('customerID', df.index)
        
        X = engineer_features(df)
        probs = pipeline.predict_proba(X)[:, 1]
        
        results = []
        for i, prob in enumerate(probs):
            will_churn = prob >= best_thresh
            results.append({
                "id": str(customer_ids[i]),
                "probability": round(float(prob), 4),
                "prediction": "Churn" if will_churn else "Retain",
                "risk": "High" if prob >= 0.70 else "Medium" if prob >= best_thresh else "Low"
            })
            
        # Sort by highest risk first
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        
        return {
            "total_processed": len(results),
            "high_risk_count": sum(1 for r in results if r["prediction"] == "Churn"),
            "data": results[:100] # Return top 100 for batch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.post("/api/send-email")
async def send_retention_email(request: EmailRequest):
    """
    Sends a real retention email using Gmail SMTP.
    Requires environment variables: SENDER_EMAIL and MAIL_APP_PASSWORD
    """
    # ── CONFIGURATION — credentials come from environment variables ────
    # Locally: set in .env file (which is in .gitignore)
    # On Vercel: set in Project Settings → Environment Variables
    SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
    APP_PASSWORD = os.environ.get("MAIL_APP_PASSWORD", "")

    if not APP_PASSWORD or APP_PASSWORD == "":
        raise HTTPException(
            status_code=500, 
            detail="Email service not configured. Please add your App Password to api/predict.py"
        )

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"ChurnSight AI <{SENDER_EMAIL}>"
        msg['To'] = request.recipient_email
        msg['Subject'] = request.subject
        msg.attach(MIMEText(request.body, 'plain'))

        # Connect and send
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        return {"status": "success", "message": f"Email sent to {request.recipient_email}"}
    
    except Exception as e:
        print(f"SMTP Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Email delivery failed: {str(e)}")


# Triggered for CodeRabbit Code Review
