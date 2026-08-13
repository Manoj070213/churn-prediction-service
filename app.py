"""FastAPI service that serves churn predictions in <100ms."""
import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("churn")

app = FastAPI(title="Churn Prediction Service")

# Load once at startup (NOT per request) -> keeps latency low
model = joblib.load("churn_model.pkl")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info(f"path={request.url.path} latency_ms={ms:.1f}")
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Customer):
    df = pd.DataFrame([c.dict()])
    prob = float(model.predict_proba(df)[:, 1][0])
    return {"churn_probability": round(prob, 4), "churn": prob > 0.5}