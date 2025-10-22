"""
01_preprocess.py
Purpose: Ingest and clean monthly U.S. marriage rate data, standardize columns, and
         write a monthly time series CSV suitable for modeling.
Input:   DATA/raw/MarriageRates.csv
Output:  DATA/processed/MarriageRates_tidy.csv  (columns: date, rate)
Run:     python SCRIPTS/01_preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("DATA/raw/MarriageRates.csv")
OUT_DIR = Path("DATA/processed")
OUT = OUT_DIR / "MarriageRates_tidy.csv"

def to_month_number(x):
    """Robust month parser: handles 1..12, 'Jan'/'January', dates."""
    if pd.isna(x):
        return np.nan
    try:
        xi = int(x)
        if 1 <= xi <= 12:
            return xi
    except Exception:
        pass
    # Try abbreviated / full month name
    for fmt in ("%b", "%B"):
        dt = pd.to_datetime(str(x), format=fmt, errors="coerce")
        if pd.notna(dt):
            return int(dt.month)
    # Try generic parse
    dt = pd.to_datetime(str(x), errors="coerce")
    if pd.notna(dt):
        return int(dt.month)
    return np.nan

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing input file: {RAW}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(RAW)

    # Normalize column names
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map to expected names
    rename_map = {
        "rate_per_1000": "rate",
        "number": "count",
        "year": "year",
        "month": "month",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Validate required columns
    required = {"year", "month", "rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["month_num"] = df["month"].apply(to_month_number)

    df = df.dropna(subset=["year", "month_num", "rate"]).copy()
    df["year"] = df["year"].astype(int)
    df["month_num"] = df["month_num"].astype(int)

    # Build monthly date (first of month)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_num"], day=1))
    df = df.sort_values("date")

    # Keep essentials; enforce monthly frequency
    ts = (
        df.set_index("date")["rate"]
        .asfreq("MS")
        .rename("rate")
        .to_frame()
        .reset_index()
    )

    # Save
    ts.to_csv(OUT, index=False)
    print(f"[OK] Wrote {len(ts)} rows to {OUT}")

if __name__ == "__main__":
    main()
