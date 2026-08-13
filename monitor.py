"""Drift monitoring: PSI on input features + auto-retrain when drift is high."""
import json
import subprocess
import numpy as np
import pandas as pd

PSI_THRESHOLD = 0.25   # >0.25 = significant drift


def psi(expected, actual, bins=10):
    """Population Stability Index between a baseline and a new sample."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected)
    a = np.histogram(actual, breakpoints)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift(new_data_path, baseline_path="baseline.json"):
    with open(baseline_path) as f:
        baseline = json.load(f)

    new = pd.read_csv(new_data_path)
    new["TotalCharges"] = pd.to_numeric(new["TotalCharges"], errors="coerce")
    new = new.dropna(subset=["TotalCharges"])

    scores = {}
    for col, ref_vals in baseline["numeric"].items():
        if col in new.columns:
            s = psi(ref_vals, new[col].dropna())
            scores[col] = s
            flag = "  <-- DRIFT" if s > PSI_THRESHOLD else ""
            print(f"PSI {col:16s}: {s:.3f}{flag}")

    max_psi = max(scores.values())
    if max_psi > PSI_THRESHOLD:
        print(f"\nSignificant drift (max PSI {max_psi:.3f}). Retraining...")
        subprocess.run(["python", "train.py"], check=True)
    else:
        print(f"\nNo significant drift (max PSI {max_psi:.3f}). No action needed.")


if __name__ == "__main__":
    check_drift("telco_churn.csv")