"""Live inference: pull the latest data from Open-Meteo and forecast the next 72 hours."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import aqi, ingest, quality
from . import config as C
from . import features as F
from . import models as M

MODEL_DIR = C.ARTIFACTS_DIR / "model"


def now_local() -> pd.Timestamp:
    """Current hour in Lahore local time (tz-naive, matching the data)."""
    return pd.Timestamp.now(tz=C.TIMEZONE).tz_localize(None).floor("h")


def build_live_frame(past_days: int = 14, origin: pd.Timestamp | None = None) -> tuple:
    """Return (frame, origin). Pollutant values after `origin` are masked out (they would be
    model forecasts, not observations); weather is kept for the whole horizon."""
    origin = origin or now_local()
    aq = ingest.fetch_air_quality_recent(past_days=past_days, forecast_days=1)
    wx = ingest.fetch_weather_forecast(past_days=past_days, forecast_days=4)
    df = wx.join(aq, how="outer")
    df = df[~df.index.duplicated()].sort_index()
    df.loc[df.index > origin, C.AQ_VARS] = np.nan
    df = quality.clean(df, max_gap_hours=6)
    return df, origin


def forecast_from_frame(df: pd.DataFrame, origin: pd.Timestamp, model_dir: Path = MODEL_DIR):
    """Forecast PM2.5 for origin+1 ... origin+72 h with a calibrated 80% band."""
    forecaster = M.Forecaster.load(model_dir)
    offsets = json.loads((model_dir / "cqr_offsets.json").read_text())
    X, meta = F.build_dataset(
        df, origins=pd.DatetimeIndex([origin]), require_target=False
    )
    if X.empty:
        raise ValueError("no usable forecast origin: check the latest observations")
    pred = M.apply_cqr(forecaster.predict(X), meta["horizon"].to_numpy(), offsets)
    out = pd.concat([meta[["origin", "target_time", "horizon"]], pred], axis=1)
    out["category"] = [aqi.category_name(v) for v in out["q50"]]
    return out


def live_forecast(past_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """(forecast table, recent observations, origin)."""
    df, origin = build_live_frame(past_days)
    fc = forecast_from_frame(df, origin)
    obs = df.loc[:origin, ["pm2_5"]].dropna().tail(24 * 7)
    return fc, obs, origin


def summarise(fc: pd.DataFrame) -> dict:
    """Headline numbers for the alert banner."""
    hi = fc.loc[fc["q50"].idxmax()]
    alert_hours = fc[aqi.is_alert(fc["q50"])]
    clean_hours = fc[fc["q50"] < aqi.BREAKPOINTS[1][0]]
    return {
        "peak_time": hi["target_time"],
        "peak_pm25": float(hi["q50"]),
        "peak_category": aqi.category_name(hi["q50"]),
        "alert_hours": int(len(alert_hours)),
        "first_alert": alert_hours["target_time"].min() if len(alert_hours) else None,
        "next_clean_window": clean_hours["target_time"].min() if len(clean_hours) else None,
    }
