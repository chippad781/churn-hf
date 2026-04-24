"""
predictor.py
------------
Shared prediction logic loaded once and reused by app.py.
Keeps model loading out of Streamlit's rerun cycle via @st.cache_resource.
"""

import os
import joblib
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP  = {
    "Electronic check": 0,
    "Mailed check":     1,
    "Bank transfer":    2,
    "Credit card":      3,
}


def load_artifacts():
    rf      = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    lr      = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    metrics = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    return rf, lr, scaler, metrics


def risk_label(prob: float) -> str:
    if prob >= 0.70: return "🔴 High"
    if prob >= 0.40: return "🟡 Medium"
    return "🟢 Low"


def predict(inputs: dict, model_choice: str, rf, lr, scaler) -> dict:
    """
    inputs keys (all required):
        tenure, monthly_charges, total_charges, num_services,
        contract_type (str), tech_support (0/1), online_security (0/1),
        senior_citizen (0/1), payment_method (str)
    model_choice: "Random Forest" | "Logistic Regression"
    """
    row = np.array([[
        inputs["tenure"],
        inputs["monthly_charges"],
        inputs["total_charges"],
        inputs["num_services"],
        CONTRACT_MAP[inputs["contract_type"]],
        inputs["tech_support"],
        inputs["online_security"],
        inputs["senior_citizen"],
        PAYMENT_MAP[inputs["payment_method"]],
    ]])

    if model_choice == "Random Forest":
        prob = float(rf.predict_proba(row)[0, 1])
    else:
        prob = float(lr.predict_proba(scaler.transform(row))[0, 1])

    pred = int(prob >= 0.5)
    return {
        "prediction":  pred,
        "label":       "⚠️ Likely to Churn" if pred else "✅ Likely to Stay",
        "probability": round(prob * 100, 1),
        "risk":        risk_label(prob),
    }
