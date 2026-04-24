"""
app.py  ←  HuggingFace Spaces entry point
-----------------------------------------
Streamlit UI for Customer Churn Prediction.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# Auto-train if model artifacts are missing (first cold start on HF Spaces)
import os
if not os.path.exists("model/rf_model.pkl"):
    import train
    train.train()

import predictor

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ── Load models (cached across reruns) ────────────────────────────────────
@st.cache_resource
def get_models():
    return predictor.load_artifacts()

rf, lr, scaler, metrics = get_models()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer is likely to churn using "
    "**Random Forest** or **Logistic Regression**."
)
st.markdown("---")

# ── Sidebar — Model info ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.subheader("Random Forest")
    st.metric("Accuracy", f"{metrics['rf_accuracy']*100:.1f}%")
    st.metric("ROC-AUC",  f"{metrics['rf_roc_auc']:.4f}")
    st.subheader("Logistic Regression")
    st.metric("Accuracy", f"{metrics['lr_accuracy']*100:.1f}%")
    st.metric("ROC-AUC",  f"{metrics['lr_roc_auc']:.4f}")
    st.markdown("---")
    st.caption(f"Trained on {metrics['dataset_size']:,} synthetic telecom records")
    st.caption(f"Dataset churn rate: {metrics['churn_rate']*100:.1f}%")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "📖 About"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Account Info**")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        contract_type = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        senior_citizen = st.radio("Senior Citizen?", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No",
                                  horizontal=True)

    with col2:
        st.markdown("**💰 Billing**")
        monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, step=0.5)
        total_charges   = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=round(float(monthly_charges * tenure), 2),
            step=10.0,
        )
        num_services = st.slider("Number of Services", 1, 6, 3)

    with col3:
        st.markdown("**🛡️ Add-ons**")
        tech_support    = st.radio("Tech Support?",    [0, 1],
                                   format_func=lambda x: "Yes" if x else "No",
                                   horizontal=True)
        online_security = st.radio("Online Security?", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No",
                                   horizontal=True)
        st.markdown(" ")
        model_choice = st.selectbox("🤖 Model", ["Random Forest", "Logistic Regression"])

    st.markdown("---")
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        result = predictor.predict(
            inputs={
                "tenure":           tenure,
                "monthly_charges":  monthly_charges,
                "total_charges":    total_charges,
                "num_services":     num_services,
                "contract_type":    contract_type,
                "tech_support":     tech_support,
                "online_security":  online_security,
                "senior_citizen":   senior_citizen,
                "payment_method":   payment_method,
            },
            model_choice=model_choice,
            rf=rf, lr=lr, scaler=scaler,
        )

        # Result display
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction",       result["label"])
        r2.metric("Churn Probability", f"{result['probability']}%")
        r3.metric("Risk Level",        result["risk"])

        # Progress bar
        prob = result["probability"] / 100
        color = "red" if prob >= 0.7 else "orange" if prob >= 0.4 else "green"
        st.markdown(f"**Churn probability: {result['probability']}%**")
        st.progress(prob)

        # Recommendation
        st.markdown("### 💡 Recommended Action")
        if prob >= 0.70:
            st.error(
                "**High churn risk.** Immediate intervention recommended:\n"
                "- Offer loyalty discount or contract upgrade\n"
                "- Personal outreach from retention team\n"
                "- Consider complimentary service add-ons"
            )
        elif prob >= 0.40:
            st.warning(
                "**Medium churn risk.** Proactive engagement suggested:\n"
                "- Send personalised retention email\n"
                "- Offer a service bundle upgrade\n"
                "- Monitor usage patterns"
            )
        else:
            st.success(
                "**Low churn risk.** Customer appears satisfied:\n"
                "- Continue standard engagement\n"
                "- Consider upsell opportunities"
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch prediction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Predict from CSV")
    st.markdown(
        "Upload a CSV with these columns: "
        "`tenure, monthly_charges, total_charges, num_services, "
        "contract_type, tech_support, online_security, senior_citizen, payment_method`"
    )

    # Sample CSV download
    sample = pd.DataFrame([
        [12,  65.5, 786.0, 3, "Month-to-month", 0, 0, 0, "Electronic check"],
        [48,  45.0, 2160.0, 5, "Two year",      1, 1, 0, "Credit card"],
        [3,   90.0, 270.0, 2, "Month-to-month", 0, 0, 1, "Mailed check"],
    ], columns=[
        "tenure","monthly_charges","total_charges","num_services",
        "contract_type","tech_support","online_security","senior_citizen","payment_method"
    ])

    csv_bytes = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV", csv_bytes,
                       "sample_customers.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    batch_model = st.selectbox("Model for batch", ["Random Forest", "Logistic Regression"],
                               key="batch_model")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** rows")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            CONTRACT_MAP = predictor.CONTRACT_MAP
            PAYMENT_MAP  = predictor.PAYMENT_MAP

            df_enc = df.copy()
            df_enc["contract_type"]  = df_enc["contract_type"].map(CONTRACT_MAP)
            df_enc["payment_method"] = df_enc["payment_method"].map(PAYMENT_MAP)

            feat_cols = ["tenure","monthly_charges","total_charges","num_services",
                         "contract_type","tech_support","online_security",
                         "senior_citizen","payment_method"]
            X = df_enc[feat_cols].values

            if batch_model == "Random Forest":
                probs = rf.predict_proba(X)[:, 1]
            else:
                probs = lr.predict_proba(scaler.transform(X))[:, 1]

            df["Churn Probability (%)"] = (probs * 100).round(1)
            df["Prediction"]            = ["Churn" if p >= 0.5 else "No Churn" for p in probs]
            df["Risk Level"]            = [predictor.risk_label(p) for p in probs]

            st.markdown("### Results")
            st.dataframe(df, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers",    len(df))
            c2.metric("Predicted Churners", int((probs >= 0.5).sum()))
            c3.metric("Churn Rate",         f"{(probs>=0.5).mean()*100:.1f}%")

            out_csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", out_csv,
                               "churn_predictions.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")
    st.markdown("""
### 🏗️ Architecture
```
┌─────────────────────────────────────────────┐
│              Streamlit UI (app.py)           │
│  ┌──────────────┐    ┌────────────────────┐  │
│  │ Single Pred  │    │  Batch Prediction  │  │
│  └──────┬───────┘    └────────┬───────────┘  │
│         └──────────┬──────────┘              │
│              predictor.py                    │
│         ┌──────────┴──────────┐              │
│    Random Forest        Logistic Regression  │
│         └──────────┬──────────┘              │
│              model/*.pkl                     │
└─────────────────────────────────────────────┘
```

### 📦 Features Used
| Feature | Description |
|---|---|
| `tenure` | Months customer has been with the company |
| `monthly_charges` | Monthly billing amount |
| `total_charges` | Total amount billed |
| `num_services` | Number of subscribed services |
| `contract_type` | Month-to-month / 1yr / 2yr |
| `tech_support` | Whether customer has tech support |
| `online_security` | Whether customer has online security |
| `senior_citizen` | Whether customer is a senior citizen |
| `payment_method` | How the customer pays |

### 🔬 Models
- **Random Forest** — 200 trees, max depth 10
- **Logistic Regression** — with StandardScaler preprocessing

### 🛠️ Tech Stack
`Python` · `scikit-learn` · `Streamlit` · `pandas` · `numpy` · `joblib`

### 🚀 Deployed on HuggingFace Spaces
""")
