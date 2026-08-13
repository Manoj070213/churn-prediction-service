"""Train the churn model and save it, plus a training baseline for drift checks."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

DATA = "telco_churn.csv"


def load_data(path=DATA):
    df = pd.read_csv(path)
    # TotalCharges loads as text with blanks -> make numeric and drop bad rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=["customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def build_pipeline(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, min_child_weight=3,
        eval_metric="logloss", random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)]), num_cols


def main():
    X, y = load_data()
    pipe, num_cols = build_pipeline(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, proba > 0.5)
    print(f"ROC-AUC:  {auc:.3f}")
    print(f"Accuracy: {acc:.3f}")

    joblib.dump(pipe, "churn_model.pkl")

    baseline = {c: X_tr[c].tolist() for c in num_cols}
    with open("baseline.json", "w") as f:
        json.dump({"numeric": baseline, "accuracy": acc}, f)
    print("Saved churn_model.pkl and baseline.json")


if __name__ == "__main__":
    main()