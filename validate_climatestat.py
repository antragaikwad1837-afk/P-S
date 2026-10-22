"""Local diagnostics for ClimateStat's CSV files, Open-Meteo API and rain model.

Run from the project folder with: python validate_climatestat.py
This script does not change your data files.
"""
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILES = {
    "india_2000_2024_daily_weather.csv": {
        "city", "date", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "rain_sum",
    },
    "Rainfall_Data_LL.csv": {"SUBDIVISION", "YEAR", "ANNUAL"},
    "rainfall_statewise_daily_imd.csv": {
        "State", "Date", "Daily Actual", "Daily Normal",
    },
}


def check_csv_files():
    print("\n[1/3] Checking project CSV files")
    all_ok = True
    for filename, required in FILES.items():
        path = DATA_DIR / filename
        if not path.is_file():
            print(f"FAIL  Missing: {path}")
            all_ok = False
            continue
        try:
            frame = pd.read_csv(path, nrows=5)
            missing = sorted(required - set(frame.columns))
            if missing:
                print(f"FAIL  {filename}: missing expected columns: {missing}")
                print(f"      Found columns: {list(frame.columns)}")
                all_ok = False
            else:
                total_rows = sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1
                print(f"PASS  {filename}: expected columns found; approx. {total_rows:,} data rows")
        except Exception as exc:
            print(f"FAIL  Cannot read {filename}: {exc}")
            all_ok = False
    return all_ok


def fetch_api_weather():
    print("\n[2/3] Checking Open-Meteo historical API")
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    end_date = today - timedelta(days=1)
    params = {
        "latitude": 18.5204,
        "longitude": 73.8567,
        "start_date": "2024-01-01",
        "end_date": end_date.isoformat(),
        "daily": [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }
    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily")
        if not daily or not daily.get("time"):
            print("FAIL  API responded but did not return daily observations")
            return None
        frame = pd.DataFrame({
            "Date": pd.to_datetime(daily["time"], errors="coerce"),
            "Mean Temperature": daily["temperature_2m_mean"],
            "Maximum Temperature": daily["temperature_2m_max"],
            "Minimum Temperature": daily["temperature_2m_min"],
            "Rainfall": daily["precipitation_sum"],
            "Wind Speed": daily["wind_speed_10m_max"],
            "Humidity": daily["relative_humidity_2m_mean"],
        }).sort_values("Date").reset_index(drop=True)
        frame = frame.dropna(subset=["Date", "Mean Temperature", "Rainfall"])
        if frame.empty:
            print("FAIL  API returned no usable rows")
            return None
        latest = frame["Date"].max().date()
        print(f"PASS  Received {len(frame):,} usable rows for Pune")
        print(f"      Requested through: {end_date}; latest returned observation: {latest}")
        print(f"      Next-day model estimate date, if features are complete: {latest + timedelta(days=1)}")
        if latest >= today:
            print("WARN  Archive returned today's or a future observation; inspect timezone/date handling")
        return frame
    except Exception as exc:
        print(f"FAIL  API request failed: {type(exc).__name__}: {exc}")
        return None


def check_model(frame):
    print("\n[3/3] Checking Logistic Regression evaluation")
    if frame is None:
        print("SKIP  No API data available, so model evaluation was not run")
        return False
    features = ["Mean Temperature", "Humidity", "Wind Speed", "Rainfall"]
    model_data = frame[["Date"] + features].copy()
    model_data["Rain Tomorrow"] = (frame["Rainfall"] > 0).astype(int)
    model_data["Previous Observation Date"] = frame["Date"].shift(1)
    for col in features:
        model_data[col] = frame[col].shift(1)
    consecutive = (model_data["Date"] - model_data["Previous Observation Date"]).dt.days.eq(1)
    model_data = model_data.loc[consecutive].dropna(subset=["Date"] + features + ["Rain Tomorrow"])
    model_data = model_data.loc[np.isfinite(model_data[features]).all(axis=1)].copy()
    model_data["Rain Tomorrow"] = model_data["Rain Tomorrow"].astype(int)
    if len(model_data) < 100 or model_data["Rain Tomorrow"].nunique() < 2:
        print(f"FAIL  Not enough usable rows/classes to evaluate model (rows={len(model_data)})")
        return False
    split = int(len(model_data) * 0.8)
    X = model_data[features]
    y = model_data["Rain Tomorrow"]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"      Usable examples: {len(model_data):,}; train: {len(X_train):,}; test: {len(X_test):,}")
    print(f"      Train dates: {model_data['Date'].iloc[0].date()} to {model_data['Date'].iloc[split-1].date()}")
    print(f"      Test dates:  {model_data['Date'].iloc[split].date()} to {model_data['Date'].iloc[-1].date()}")
    print(f"      Rain class share: train={y_train.mean():.1%}, test={y_test.mean():.1%}")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("FAIL  Train/test period has only one class; ROC-AUC and model metrics are not reliable")
        return False
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print("PASS  Model trained and evaluated without an exception")
    print(f"      Accuracy={accuracy_score(y_test, pred):.3f}; Precision={precision_score(y_test, pred, zero_division=0):.3f}; Recall={recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"      F1={f1_score(y_test, pred, zero_division=0):.3f}; Balanced accuracy={balanced_accuracy_score(y_test, pred):.3f}; ROC-AUC={roc_auc_score(y_test, prob):.3f}")
    live = frame.dropna(subset=features).copy()
    live = live.loc[np.isfinite(live[features]).all(axis=1)]
    if live.empty:
        print("FAIL  No latest row has all required features for prediction")
        return False
    last = live.iloc[-1]
    next_date = last["Date"].date() + timedelta(days=1)
    p = model.predict_proba(live[features].iloc[[-1]])[0, 1]
    print(f"PASS  Inference succeeded: estimated rain probability={p:.1%} for {next_date}")
    print("NOTE  This is a historical-data-based model estimate, not an official weather forecast.")
    return True


if __name__ == "__main__":
    csv_ok = check_csv_files()
    api_frame = fetch_api_weather()
    model_ok = check_model(api_frame)
    if csv_ok and api_frame is not None and model_ok:
        print("\nOVERALL: CSV schema, API, and model checks passed in this environment.")
        sys.exit(0)
    print("\nOVERALL: One or more checks failed or could not run. Review the FAIL/SKIP lines above.")
    sys.exit(1)
