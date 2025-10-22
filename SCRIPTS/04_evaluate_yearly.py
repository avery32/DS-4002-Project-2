#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_evaluate_yearly.py
Purpose:
    Aggregate monthly forecasts to annual means and evaluate against
    actual CDC marriage rates provided in an external CSV file.

Inputs:
    - DATA/processed/MarriageRates_tidy.csv
    - OUTPUT/forecast_2010_2023.csv      (monthly predictions from SARIMA)
    - DATA/2020sMarriageRates.csv        (actual yearly marriage data uploaded by user)

Outputs:
    - OUTPUT/forecast_vs_actual_yearly.csv
    - OUTPUT/metrics_yearly.json

Run:
    python SCRIPTS/04_evaluate_yearly.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Paths ----------
IN_FORE    = Path("OUTPUT/forecast_2010_2023.csv")  # from your forecast script
IN_ACTUAL  = Path("DATA/2020sMarriageRates.csv")    # uploaded file
OUT_DIR    = Path("OUTPUT")
OUT_COMPARE = OUT_DIR / "forecast_vs_actual_yearly.csv"
OUT_METRICS = OUT_DIR / "metrics_yearly.json"

# ---------- Helpers ----------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model forecast (monthly) ----
    if not IN_FORE.exists():
        raise FileNotFoundError(f"Missing forecast file: {IN_FORE}")
    df_fc = pd.read_csv(IN_FORE, parse_dates=["date"])
    if "pred" not in df_fc.columns:
        raise ValueError(f"Forecast file {IN_FORE} must have a 'pred' column.")
    df_fc["year"] = df_fc["date"].dt.year
    forecast_yearly = df_fc.groupby("year", as_index=True)["pred"].mean().rename("pred_rate")

    # ---- Load actual annual data ----
    if not IN_ACTUAL.exists():
        raise FileNotFoundError(f"Missing actual data CSV: {IN_ACTUAL}")
    df_actual = pd.read_csv(IN_ACTUAL)

    # Try to normalize column names
    df_actual.columns = [c.strip().lower().replace(" ", "_") for c in df_actual.columns]

    # Expecting columns like: year, rate_per_1000_total
    possible_rate_cols = [c for c in df_actual.columns if "rate" in c]
    if not possible_rate_cols:
        raise ValueError(f"Could not find a 'rate' column in {IN_ACTUAL}. Found: {list(df_actual.columns)}")
    rate_col = possible_rate_cols[0]

    actual_yearly = (
        df_actual
        .dropna(subset=["year", rate_col])
        .set_index("year")[rate_col]
        .astype(float)
        .rename("actual_rate")
    )

    # ---- Merge + compare overlapping years ----
    compare = pd.concat([forecast_yearly, actual_yearly], axis=1).dropna()
    if compare.empty:
        raise ValueError("No overlapping years between forecast and actual data. "
                         "Extend forecast horizon if needed (e.g., through 2023).")

    # ---- Compute errors ----
    compare["abs_error"] = (compare["actual_rate"] - compare["pred_rate"]).abs()
    compare["abs_perc_error"] = 100.0 * compare["abs_error"] / compare["actual_rate"]

    y_true = compare["actual_rate"].values
    y_pred = compare["pred_rate"].values
    metrics = {
        "years_compared": list(map(int, compare.index.tolist())),
        "n_years": int(len(compare)),
        "MAPE_percent": mape(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(compare) >= 2 else None,
        "actual_annual_mean": float(np.mean(y_true)),
        "pred_annual_mean": float(np.mean(y_pred)),
    }

    # ---- Save outputs ----
    compare.reset_index(names="Year").to_csv(OUT_COMPARE, index=False)
    with OUT_METRICS.open("w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Console summary ----
    print(f"[OK] Wrote yearly comparison -> {OUT_COMPARE}")
    print(f"[OK] Wrote yearly metrics   -> {OUT_METRICS}")
    print(compare)
    print("\nMetrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
