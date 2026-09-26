"""Generate report figures and small summary artifacts from a finished training run.

Usage:  python -m aqforecast.report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap

from . import config as C
from . import features as F
from . import ingest, quality, viz
from . import models as M


def main() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = quality.clean(ingest.load_history())
    df = df.loc[df["pm2_5"].first_valid_index() :]
    metrics = json.loads((C.ARTIFACTS_DIR / "metrics.json").read_text())
    pred = pd.read_parquet(C.PROCESSED_DIR / "test_predictions.parquet")

    # Daily history for the app's trends tab (refreshed daily by aqforecast.refresh)
    daily = df["pm2_5"].dropna().resample("D").agg(["mean", "max"]).round(1)
    daily = daily[daily.index < daily.index.max()]
    daily.to_parquet(C.ARTIFACTS_DIR / "daily_pm25.parquet")

    viz.plot_seasonality(df, C.FIGURES_DIR / "01_seasonality.png")
    viz.plot_drivers(df, C.FIGURES_DIR / "02_weather_drivers.png")
    by_h = {**metrics["by_horizon"]}
    viz.plot_mae_by_horizon(by_h, C.FIGURES_DIR / "03_mae_by_horizon.png")
    viz.plot_episode(pred, C.FIGURES_DIR / "04_smog_episode.png")
    viz.plot_confusion(pred.y, pred.lgbm_q50, C.FIGURES_DIR / "05_alert_confusion.png")
    viz.plot_coverage(pred, C.FIGURES_DIR / "06_interval_calibration.png")

    # SHAP on the hold-out model, 13-24h bucket (median model), sample of hold-out rows
    forecaster = M.Forecaster.load(C.MODELS_DIR / "holdout")
    X, meta = F.build_dataset(df, horizons=range(13, 25), origin_step=12)
    test_start = pd.Timestamp(metrics["data"]["holdout_start"])
    X = X[(meta["origin"] >= test_start).to_numpy()].reset_index(drop=True)
    sample = X.sample(min(3000, len(X)), random_state=C.SEED)[forecaster.feature_names]
    booster = forecaster.models[(3, 0.5)]
    sv = shap.TreeExplainer(booster).shap_values(sample)
    viz.plot_shap(sv, sample, C.FIGURES_DIR / "07_shap_importance.png")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    (C.ARTIFACTS_DIR / "shap_top.json").write_text(json.dumps(imp.head(15).round(4).to_dict(),
                                                              indent=1))
    print(imp.head(10))
    print("figures written to", C.FIGURES_DIR)


if __name__ == "__main__":
    main()
