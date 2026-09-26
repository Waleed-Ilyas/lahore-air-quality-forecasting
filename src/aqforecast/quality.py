"""Data-quality checks: schema, ranges, duplicates, gaps and missing values."""
from __future__ import annotations

import pandas as pd

from . import config as C

# Physically plausible ranges (min, max). Values outside are flagged and cleaned.
RANGES = {
    "pm2_5": (0, 1000), "pm10": (0, 2000), "carbon_monoxide": (0, 50000),
    "nitrogen_dioxide": (0, 1000), "sulphur_dioxide": (0, 2000), "ozone": (0, 1000),
    "aerosol_optical_depth": (0, 10), "dust": (0, 5000),
    "temperature_2m": (-20, 60), "relative_humidity_2m": (0, 100), "dew_point_2m": (-30, 45),
    "precipitation": (0, 200), "surface_pressure": (850, 1100), "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 150), "wind_direction_10m": (0, 360), "boundary_layer_height": (0, 6000),
}


def check(df: pd.DataFrame) -> dict:
    """Return a report dict. Raises ValueError on hard failures (schema / index order)."""
    expected = set(C.AQ_VARS + C.WEATHER_VARS)
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing columns: {sorted(missing_cols)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted by time")

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    report = {
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_hours": int(len(full.difference(df.index))),
        "missing_pct": (df[sorted(expected)].isna().mean() * 100).round(3).to_dict(),
        "out_of_range": {},
    }
    for col, (lo, hi) in RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            report["out_of_range"][col] = bad
    return report


def clean(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """Reindex to a complete hourly grid, null impossible values, interpolate short gaps."""
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="time"))
    for col, (lo, hi) in RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = float("nan")
    # Only short gaps are interpolated; long gaps stay NaN so those rows can be dropped honestly.
    return df.interpolate(method="time", limit=max_gap_hours, limit_area="inside")
