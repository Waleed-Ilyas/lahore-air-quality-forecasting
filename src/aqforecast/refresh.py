"""Daily refresh job (run by GitHub Actions): fetch fresh data, forecast, log, track live accuracy.

Writes small files into artifacts/ so the deployed app always has a recent snapshot:
  latest_forecast.parquet   the 72h forecast made now
  latest_obs.parquet        the last 7 days of observed PM2.5
  forecast_log.parquet      every forecast issued in the last 120 days
  live_accuracy.json        MAE of past forecasts against what was actually observed
  daily_pm25.parquet        daily mean / max PM2.5 history for the trends tab
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import inference

LOG_DAYS = 120


def update_daily_history(df: pd.DataFrame) -> None:
    path = C.ARTIFACTS_DIR / "daily_pm25.parquet"
    new = df["pm2_5"].dropna().resample("D").agg(["mean", "max"]).round(1)
    new = new[new.index < new.index.max()]  # drop the (partial) current day
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    new.to_parquet(path)


def update_log_and_accuracy(fc: pd.DataFrame, df: pd.DataFrame) -> dict:
    path = C.ARTIFACTS_DIR / "forecast_log.parquet"
    cols = ["origin", "target_time", "horizon", "q10", "q50", "q90"]
    log = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=cols)
    log = pd.concat([log[~log["origin"].isin(fc["origin"])], fc[cols]], ignore_index=True)
    log = log.astype({"origin": "datetime64[ns]", "target_time": "datetime64[ns]", "horizon": "int64"})
    log = log[log["origin"] >= log["origin"].max() - pd.Timedelta(days=LOG_DAYS)]
    log.to_parquet(path)

    obs = df["pm2_5"].dropna().rename("actual")
    obs.index = obs.index.astype("datetime64[ns]")
    joined = log.join(obs, on="target_time").dropna(subset=["actual"])
    joined = joined[joined["target_time"] <= fc["origin"].iloc[0]]  # only realised targets
    if joined.empty:
        return {"n": 0}
    err = (joined["q50"] - joined["actual"]).abs()
    buckets = {"1-24h": (1, 24), "25-48h": (25, 48), "49-72h": (49, 72)}
    out = {"n": int(len(joined)), "mae_all": float(err.mean()),
           "coverage_pct": float(((joined.actual >= joined.q10) & (joined.actual <= joined.q90)).mean() * 100),
           "since": str(joined["origin"].min())}
    for name, (lo, hi) in buckets.items():
        m = (joined["horizon"] >= lo) & (joined["horizon"] <= hi)
        out[f"mae_{name}"] = float(err[m].mean()) if m.any() else None
    return out


def main() -> None:
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    df, origin = inference.build_live_frame(past_days=14)
    fc = inference.forecast_from_frame(df, origin)
    fc.to_parquet(C.ARTIFACTS_DIR / "latest_forecast.parquet")
    df.loc[:origin, ["pm2_5"]].dropna().tail(24 * 7).to_parquet(C.ARTIFACTS_DIR / "latest_obs.parquet")
    update_daily_history(df)
    acc = update_log_and_accuracy(fc, df)
    (C.ARTIFACTS_DIR / "live_accuracy.json").write_text(json.dumps(acc, indent=1))
    peak = fc.loc[fc["q50"].idxmax()]
    print(f"origin {origin}: peak {peak.q50:.0f} ug/m3 at {peak.target_time}; live accuracy {acc}")
    assert np.isfinite(fc[["q10", "q50", "q90"]].to_numpy()).all()


if __name__ == "__main__":
    main()
