"""
02_model.py
Purpose: Simple AIC-based model selection over small ARIMA/SARIMA grids.
Input:   DATA/processed/MarriageRates_tidy.csv
Output:  OUTPUT/model_selection_results.csv
         OUTPUT/best_model.json  (fields: family, order, seasonal_order)
Run:     python SCRIPTS/02_model.py
Notes:   Keeps grids modest so it runs fast. Seasonality assumes monthly data (s=12).
"""

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

IN_FILE = Path("DATA/processed/MarriageRates_tidy.csv")
OUT_DIR = Path("OUTPUT")
SEL_CSV = OUT_DIR / "model_selection_results.csv"
BEST_JSON = OUT_DIR / "best_model.json"

# Training window: 2000–2009 (per your plan)
TRAIN_START, TRAIN_END = 2000, 2009

def load_series():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing processed file: {IN_FILE}")
    df = pd.read_csv(IN_FILE, parse_dates=["date"])
    ts = df.set_index("date")["rate"].asfreq("MS")
    train = ts[(ts.index.year >= TRAIN_START) & (ts.index.year <= TRAIN_END)].dropna()
    if train.empty:
        raise ValueError("Training slice (2000–2009) is empty. Check your data.")
    return train

def try_fit_arima(y, order):
    try:
        res = ARIMA(y, order=order).fit()
        return res.aic
    except Exception:
        return np.inf

def try_fit_sarima(y, order, seasonal_order):
    try:
        res = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return res.aic
    except Exception:
        return np.inf

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_series()

    results = []

    # Small ARIMA grid
    arima_grid = [(p,d,q) for p in (0,1,2) for d in (0,1,2) for q in (0,1,2)]
    for order in arima_grid:
        aic = try_fit_arima(y, order)
        results.append({"family":"ARIMA", "order":order, "seasonal_order":None, "aic":aic})

    # Small SARIMA grid (s=12)
    s = 12
    pdq = [(p,d,q) for p in (0,1,2) for d in (0,1) for q in (0,1,2)]
    PDQ = [(P,D,Q,s) for P in (0,1) for D in (0,1) for Q in (0,1)]
    for order in pdq:
        for seasonal_order in PDQ:
            aic = try_fit_sarima(y, order, seasonal_order)
            results.append({"family":"SARIMA", "order":order, "seasonal_order":seasonal_order, "aic":aic})

    df_res = pd.DataFrame(results).sort_values("aic", ascending=True)
    df_res.to_csv(SEL_CSV, index=False)
    print(f"[OK] wrote {SEL_CSV} (top 5):")
    print(df_res.head(5))

    best = df_res.iloc[0].to_dict()
    # Convert tuples to lists for JSON
    best["order"] = list(best["order"]) if best["order"] is not None else None
    best["seasonal_order"] = list(best["seasonal_order"]) if isinstance(best["seasonal_order"], (tuple,list)) else None

    with BEST_JSON.open("w") as f:
        json.dump(best, f, indent=2)
    print(f"[OK] best model -> {BEST_JSON}")

if __name__ == "__main__":
    main()
