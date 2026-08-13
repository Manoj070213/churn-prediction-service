# Churn Prediction Service

A customer churn prediction model served as a REST API, with drift monitoring and automated retraining. Built on the Telco Customer Churn dataset (7,043 records).

## Performance
- ROC-AUC: ~0.84
- Accuracy: ~0.80

## Tech Stack
Python, scikit-learn, XGBoost, FastAPI, pandas

## Files
- `train.py` — trains the model, saves the pipeline and a drift baseline
- `app.py` — FastAPI service exposing `/predict` and `/health` endpoints
- `monitor.py` — PSI-based drift detection with automatic retraining
- `requirements.txt` — dependencies
- `telco_churn.csv` — the dataset

## Setup

    pip install -r requirements.txt

## Train the model

    python train.py

## Run the API

    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs to test predictions interactively.

## Monitor for drift

    python monitor.py

Computes the Population Stability Index (PSI) for each numeric feature and retrains automatically if drift crosses the threshold.
