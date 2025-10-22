"""
03_fit_and_forecast.py (robust start/end handling)
Purpose: Fit the selected model on 2000–2009 monthly data and forecast through Dec 2021.
Input:   DATA/processed/MarriageRates_tidy.csv
         OUTPUT/best_model.json (optional) - if missing, falls back to SARIMA(1,1,1)x(1,1,1,12)
Output:  OUTPUT/forecast_2010_2023.csv  (date, pred)
Run:     python SCRIPTS/03_fit_and_forecast.py
"""

import json
from pathlib import Path
import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

IN_FILE  = Path("DATA/processed/MarriageRates_tidy.csv")
BEST_JSON = Path("OUTPUT/best_model.json")
OUT_DIR  = Path("OUTPUT")
OUT_FC   = OUT_DIR / "forecast_2010_2023.csv"

TRAIN_START, TRAIN_END = 2000, 2009
FC_END = pd.Timestamp("2023-12-01")  # monthly start

def load_series():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing processed file: {IN_FILE}")
    df = pd.read_csv(IN_FILE, parse_dates=["date"])
    ts = df.set_index("date")["rate"].asfreq("MS")
    return ts

def load_best():
    if BEST_JSON.exists():
        return json.loads(BEST_JSON.read_text())
    # Fallback default
    return {"family": "SARIMA", "order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]}

def months_between_inclusive(start_month: pd.Timestamp, end_month: pd.Timestamp) -> int:
    """Number of months from start_month+1 month to end_month inclusive
       (i.e., forecast months AFTER the training sample)."""
    # We want forecasts *after* the training last month:
    # last_train ... [forecast starts next month] ... FC_END
    total = (end_month.year - start_month.year) * 12 + (end_month.month - start_month.month)
    if total < 1:
        return 0
    return total  # because we start at the next month already

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = load_series()
    # Keep monthly freq on the slice too
    train = ts[(ts.index.year >= TRAIN_START) & (ts.index.year <= TRAIN_END)].asfreq("MS").dropna()
    if train.empty:
        raise ValueError("Training slice (2000–2009) is empty. Check your data.")

    best = load_best()
    fam = (best.get("family") or "SARIMA").upper()
    order = tuple(best.get("order", [1, 1, 1]))
    seasonal_order = best.get("seasonal_order", None)
    if fam == "SARIMA":
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 12)
        else:
            seasonal_order = tuple(seasonal_order)

    last_train = train.index[-1]                 # e.g., 2009-12-01
    steps = months_between_inclusive(last_train, FC_END)  # months to forecast
    if steps <= 0:
        raise ValueError("Nothing to forecast: FC_END is not after the training end.")

    # Fit
    if fam == "ARIMA":
        res = ARIMA(train, order=order).fit()
    else:
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    # Forecast without relying on timestamp indexing
    fc_res = res.get_forecast(steps=steps)
    pred_mean = fc_res.predicted_mean  # length == steps

    # Build the forecast index manually: next month after training through FC_END
    fc_index = pd.date_range(start=last_train + pd.offsets.MonthBegin(1),
                             periods=steps, freq="MS")
    out = (pd.Series(pred_mean.values, index=fc_index, name="pred")
           .to_frame().reset_index().rename(columns={"index": "date"}))

    out.to_csv(OUT_FC, index=False)
    print(f"[OK] wrote forecast ({len(out)} rows) to {OUT_FC}")

if __name__ == "__main__":
    main()
