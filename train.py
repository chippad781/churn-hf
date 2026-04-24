"""
train.py
--------
Generates synthetic telecom churn data, trains Random Forest +
Logistic Regression models, and saves all artifacts to ./model/

Run once before launching the app:
    python train.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

SEED = 42
np.random.seed(SEED)


# ── 1. Synthetic dataset ────────────────────────────────────────────────────
def generate_data(n: int = 5000) -> pd.DataFrame:
    tenure          = np.random.randint(1, 73, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    num_services    = np.random.randint(1, 7, n)
    contract_type   = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # 0=M2M,1=1yr,2=2yr
    tech_support    = np.random.choice([0, 1], n)
    online_security = np.random.choice([0, 1], n)
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    payment_method  = np.random.choice([0, 1, 2, 3], n)

    churn_prob = np.clip(
        0.30
        - 0.004 * tenure
        + 0.003 * monthly_charges
        - 0.080 * contract_type
        - 0.050 * tech_support
        - 0.040 * online_security
        + 0.050 * senior_citizen
        + np.random.normal(0, 0.05, n),
        0.03, 0.95,
    )
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "num_services":     num_services,
        "contract_type":    contract_type,
        "tech_support":     tech_support,
        "online_security":  online_security,
        "senior_citizen":   senior_citizen,
        "payment_method":   payment_method,
        "churn":            churn,
    })


# ── 2. Train & save ─────────────────────────────────────────────────────────
def train():
    print("Generating data …")
    df = generate_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/churn_data.csv", index=False)
    print(f"  Rows: {len(df)}  |  Churn rate: {df['churn'].mean():.2%}")

    X, y = df.drop("churn", axis=1), df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred  = rf.predict(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_pred  = lr.predict(X_test_sc)

    print("\n── Random Forest ──────────────────────")
    print(f"  Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, rf_proba):.4f}")
    print(classification_report(y_test, rf_pred, target_names=["No Churn", "Churn"]))

    print("── Logistic Regression ────────────────")
    print(f"  Accuracy : {accuracy_score(y_test, lr_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, lr_proba):.4f}")
    print(classification_report(y_test, lr_pred, target_names=["No Churn", "Churn"]))

    metrics = {
        "rf_accuracy":  round(accuracy_score(y_test, rf_pred), 4),
        "rf_roc_auc":   round(roc_auc_score(y_test, rf_proba), 4),
        "lr_accuracy":  round(accuracy_score(y_test, lr_pred), 4),
        "lr_roc_auc":   round(roc_auc_score(y_test, lr_proba), 4),
        "churn_rate":   round(float(df["churn"].mean()), 4),
        "dataset_size": len(df),
    }

    os.makedirs("model", exist_ok=True)
    joblib.dump(rf,             "model/rf_model.pkl")
    joblib.dump(lr,             "model/lr_model.pkl")
    joblib.dump(scaler,         "model/scaler.pkl")
    joblib.dump(list(X.columns),"model/feature_names.pkl")
    joblib.dump(metrics,        "model/metrics.pkl")
    print("\n✅  Saved to model/")


if __name__ == "__main__":
    train()
