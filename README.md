---
title: Customer Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 📉 Customer Churn Prediction System

An end-to-end ML web application that predicts whether a telecom customer is likely to churn, built with **scikit-learn** and **Streamlit**.

## 🏗️ Architecture

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

## 📁 Project Structure

```
churn-prediction/
├── app.py              ← Streamlit UI (HF entry point)
├── predictor.py        ← Shared prediction logic
├── train.py            ← Data generation + model training
├── requirements.txt    ← Dependencies
├── model/
│   ├── rf_model.pkl    ← Trained Random Forest
│   ├── lr_model.pkl    ← Trained Logistic Regression
│   ├── scaler.pkl      ← StandardScaler for LR
│   ├── feature_names.pkl
│   └── metrics.pkl     ← Evaluation metrics
└── data/
    └── churn_data.csv  ← Generated training data
```

## 🔬 Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | ~73% | ~0.65 |
| Logistic Regression | ~74% | ~0.69 |

## 📦 Features

- **Single prediction** — fill in customer details and get instant churn risk
- **Batch prediction** — upload a CSV, get results + downloadable output
- **Risk level** — Low / Medium / High with recommended retention actions
- **Model metrics** — displayed live in sidebar

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `pandas` · `numpy` · `joblib`

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python train.py

# Launch app
streamlit run app.py
```

## ☁️ Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Upload all files (or push via git)
4. The app auto-trains on first cold start — no manual step needed

---
Built by [Your Name] · [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
