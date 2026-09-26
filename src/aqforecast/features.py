"""Feature engineering for direct multi-horizon PM2.5 forecasting.

One training row = (forecast origin t, horizon h). Everything in the row is known at time t,
except the *target-time weather* (weather at t+h), which is known in production from the weather
forecast. In training we use the ERA5 weather at t+h as a stand-in for that forecast; this is a
mild optimistic bias that is documented in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
ROLL_WINDOWS = (6, 24, 72, 168)
MIN_HISTORY = 168  # hours of history needed for the longest lag / window
POLLUTANTS_NOW = [
    "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "aerosol_optical_depth", "dust",
    "sulphur_dioxide",
]
WEATHER_NOW = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "boundary_layer_height"]
WEATHER_TARGET = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "boundary_layer_height", "wind_u", "wind_v",
]


def _add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    rad = np.deg2rad(df["wind_direction_10m"])
    df = df.copy()
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    return df


def origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at each origin time t (lags, rolling stats, current conditions)."""
    df = _add_wind_components(df)
    pm = df["pm2_5"]
    f = pd.DataFrame(index=df.index)
    f["pm25_now"] = pm
    for lag in LAGS:
        f[f"pm25_lag{lag}"] = pm.shift(lag)
    for w in ROLL_WINDOWS:
        f[f"pm25_mean{w}"] = pm.rolling(w, min_periods=w // 2).mean()
    f["pm25_std24"] = pm.rolling(24, min_periods=12).std()
    f["pm25_max24"] = pm.rolling(24, min_periods=12).max()
    f["pm25_min24"] = pm.rolling(24, min_periods=12).min()
    f["pm25_trend6"] = pm - pm.shift(6)
    f["pm25_trend24"] = pm - pm.shift(24)
    for col in POLLUTANTS_NOW:
        f[f"{col}_now"] = df[col]
    f["pm_coarse_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
    for col in WEATHER_NOW:
        f[f"{col}_now"] = df[col]
    f["wind_u_now"], f["wind_v_now"] = df["wind_u"], df["wind_v"]
    f["precip_sum24"] = df["precipitation"].rolling(24, min_periods=12).sum()
    f["temp_change24"] = df["temperature_2m"] - df["temperature_2m"].shift(24)
    return f


def build_dataset(
    df: pd.DataFrame,
    horizons: range | list[int] = range(1, C.HORIZON + 1),
    origin_step: int = 1,
    origins: pd.DatetimeIndex | None = None,
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the supervised (origin, horizon) table.

    `df` must be a complete hourly frame. For live inference it may contain extra *future* rows
    holding only weather (pm2_5 = NaN). Returns (X, meta); meta has origin, target time, y and
    the seasonal-naive / persistence baselines.
    """
    df = _add_wind_components(df)
    base = origin_features(df)
    pm = df["pm2_5"]
    n = len(df)

    if origins is None:
        pos = np.arange(MIN_HISTORY, n, origin_step)
    else:
        pos = df.index.get_indexer(origins)
        pos = pos[pos >= 0]
    keep = base["pm25_now"].to_numpy()[pos]
    pos = pos[~np.isnan(keep)]

    base_rows = base.iloc[pos]
    origin_times = df.index[pos]
    precip = df["precipitation"].fillna(0.0)
    precip_cum = precip.cumsum()
    wind_cum = df["wind_speed_10m"].fillna(df["wind_speed_10m"].mean()).cumsum()

    parts_x, parts_meta = [], []
    for h in horizons:
        tgt_time = origin_times + pd.Timedelta(hours=h)
        x = base_rows.copy()
        x["horizon"] = h
        fut = df[WEATHER_TARGET].shift(-h).iloc[pos]
        for col in WEATHER_TARGET:
            x[f"{col}_tgt"] = fut[col].to_numpy()
        x["precip_cum_h"] = (precip_cum.shift(-h) - precip_cum).iloc[pos].to_numpy()
        x["wind_mean_h"] = ((wind_cum.shift(-h) - wind_cum) / h).iloc[pos].to_numpy()
        x["temp_diff_tgt"] = x["temperature_2m_tgt"] - x["temperature_2m_now"]
        x["blh_diff_tgt"] = x["boundary_layer_height_tgt"] - x["boundary_layer_height_now"]
        hour = tgt_time.hour + tgt_time.minute / 60
        doy = tgt_time.dayofyear
        x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        x["month"] = tgt_time.month
        x["dayofweek"] = tgt_time.dayofweek

        k = -(-h // 24)  # ceil(h / 24)
        snaive_lag = 24 * k - h  # same clock hour as the target, on the latest fully observed day
        s1, s2, s3 = (pm.shift(snaive_lag + 24 * i).iloc[pos].to_numpy() for i in range(3))
        x["snaive_1d"] = s1
        x["snaive_3d_mean"] = np.nanmean(np.vstack([s1, s2, s3]), axis=0)

        meta = pd.DataFrame(
            {
                "origin": origin_times,
                "target_time": tgt_time,
                "horizon": h,
                "y": pm.shift(-h).iloc[pos].to_numpy(),
                "persistence": x["pm25_now"].to_numpy(),
                "snaive": s1,
                "snaive_3d": x["snaive_3d_mean"].to_numpy(),
            }
        )
        parts_x.append(x)
        parts_meta.append(meta)

    X = pd.concat(parts_x, ignore_index=True).astype("float32")
    meta = pd.concat(parts_meta, ignore_index=True)
    if require_target:
        ok = meta["y"].notna().to_numpy()
        X, meta = X[ok].reset_index(drop=True), meta[ok].reset_index(drop=True)
    return X, meta


FEATURE_NAMES: list[str] | None = None


def feature_names() -> list[str]:
    """Names of the model input columns (built once from a tiny synthetic frame)."""
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        idx = pd.date_range("2024-01-01", periods=MIN_HISTORY + 80, freq="h")
        rng = np.random.default_rng(0)
        cols = C.AQ_VARS + C.WEATHER_VARS
        demo = pd.DataFrame(rng.random((len(idx), len(cols))) + 1, index=idx, columns=cols)
        x, _ = build_dataset(demo, horizons=[1], origin_step=50, require_target=False)
        FEATURE_NAMES = list(x.columns)
    return FEATURE_NAMES
