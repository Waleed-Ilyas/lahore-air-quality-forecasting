"""Forecast metrics: point accuracy, per-horizon breakdown, AQI-category and alert skill."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from . import aqi

HORIZON_BUCKETS = {"1-6h": (1, 6), "7-24h": (7, 24), "25-48h": (25, 48), "49-72h": (49, 72)}


def point_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    err = pred - y
    ss_res, ss_tot = float((err**2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape_pct": float(np.abs(err).sum() / np.abs(y).sum() * 100),
        "bias": float(err.mean()),
        "r2": 1 - ss_res / ss_tot,
    }


def by_horizon(meta: pd.DataFrame, pred, buckets=HORIZON_BUCKETS) -> dict:
    out = {}
    pred = np.asarray(pred, float)
    for name, (lo, hi) in buckets.items():
        m = ((meta["horizon"] >= lo) & (meta["horizon"] <= hi)).to_numpy()
        out[name] = point_metrics(meta["y"].to_numpy()[m], pred[m])
    return out


def alert_metrics(y, pred) -> dict:
    """Skill at predicting 'Unhealthy or worse' hours (PM2.5 >= 55.5) and the AQI category."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ya, pa = aqi.is_alert(y), aqi.is_alert(pred)
    cy, cp = aqi.category_index(y), aqi.category_index(pred)
    return {
        "alert_base_rate_pct": float(ya.mean() * 100),
        "alert_precision": float(precision_score(ya, pa, zero_division=0)),
        "alert_recall": float(recall_score(ya, pa, zero_division=0)),
        "alert_f1": float(f1_score(ya, pa, zero_division=0)),
        "category_accuracy": float((cy == cp).mean()),
        "category_within_one": float((np.abs(cy - cp) <= 1).mean()),
    }


def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return {
        "coverage_pct": float(((y >= lo) & (y <= hi)).mean() * 100),
        "mean_width": float((hi - lo).mean()),
    }
