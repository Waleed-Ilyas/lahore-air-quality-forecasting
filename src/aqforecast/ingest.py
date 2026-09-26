"""Reproducible data ingestion from the free Open-Meteo APIs (no API key needed).

Usage:  python -m aqforecast.ingest   (full history -> data/raw/history.parquet)

Open-Meteo Air Quality data is CAMS model output (reanalysis + analysis), NOT ground-sensor readings.
Licence: CC BY 4.0 (https://open-meteo.com/en/license).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import config as C

ARCHIVE_LAG_DAYS = 6  # ERA5 archive lags real time by ~5 days; the gap comes from the forecast API


def _get(url: str, params: dict, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # network / 429 / 5xx
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Open-Meteo request failed: {url}") from last


def _to_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def _base_params(variables: list[str]) -> dict:
    return {
        "latitude": C.LAT,
        "longitude": C.LON,
        "hourly": ",".join(variables),
        "timezone": C.TIMEZONE,
    }


def fetch_air_quality(start: str, end: str) -> pd.DataFrame:
    """Hourly pollutants between two ISO dates (inclusive)."""
    params = _base_params(C.AQ_VARS) | {"start_date": start, "end_date": end}
    return _to_frame(_get(C.AQ_URL, params))


def fetch_air_quality_recent(past_days: int = 14, forecast_days: int = 1) -> pd.DataFrame:
    """Most recent hours for live inference."""
    params = _base_params(C.AQ_VARS) | {"past_days": past_days, "forecast_days": forecast_days}
    return _to_frame(_get(C.AQ_URL, params))


def fetch_weather_forecast(past_days: int = 3, forecast_days: int = 4) -> pd.DataFrame:
    """Weather forecast (future hours) plus a few past days."""
    params = _base_params(C.WEATHER_VARS) | {"past_days": past_days, "forecast_days": forecast_days}
    return _to_frame(_get(C.WEATHER_FORECAST_URL, params))


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Hourly weather: ERA5 archive where available, forecast-API analysis for the last days."""
    today = date.today()
    archive_end = min(date.fromisoformat(end), today - timedelta(days=ARCHIVE_LAG_DAYS))
    frames = []
    if date.fromisoformat(start) <= archive_end:
        params = _base_params(C.WEATHER_VARS) | {
            "start_date": start,
            "end_date": archive_end.isoformat(),
        }
        frames.append(_to_frame(_get(C.WEATHER_ARCHIVE_URL, params)))
    if date.fromisoformat(end) > archive_end:
        gap_days = (today - archive_end).days + 1
        recent = fetch_weather_forecast(past_days=min(gap_days, 92), forecast_days=1)
        cutoff = pd.Timestamp(archive_end) + pd.Timedelta(hours=23)
        frames.append(recent[recent.index > cutoff])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def download_history(end: str | None = None) -> pd.DataFrame:
    """Download the full modelling history and write it to data/raw/history.parquet."""
    end = end or date.today().isoformat()
    aq = fetch_air_quality(C.AQ_START, end)
    weather = fetch_weather(C.AQ_START, end)
    df = aq.join(weather, how="left")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.RAW_DIR / "history.parquet")
    return df


def load_history() -> pd.DataFrame:
    path = C.RAW_DIR / "history.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m aqforecast.ingest` first")
    return pd.read_parquet(path)


if __name__ == "__main__":
    data = download_history()
    print(f"Saved {len(data):,} hourly rows: {data.index.min()} -> {data.index.max()}")
