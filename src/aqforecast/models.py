"""Models: LightGBM quantile forecaster, statistical (Holt-Winters) model and simple baselines."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 400,
}


def _params(alpha: float, overrides: dict | None) -> dict:
    p = DEFAULT_PARAMS | (overrides or {})
    return p | {
        "objective": "quantile", "alpha": alpha, "random_state": C.SEED, "verbose": -1,
        "n_jobs": -1,
    }


# One model set per horizon bucket: a single global model blurs short and long horizons together
# (1h-ahead MAE was ~3x worse than a model trained on short horizons only in my experiments).
BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 72)]


def _bucket_of(horizon: np.ndarray) -> np.ndarray:
    bounds = np.array([hi for _, hi in BUCKETS])
    return np.searchsorted(bounds, horizon, side="left")


class Forecaster:
    """Direct multi-horizon quantile forecaster.

    Predicts the *change* in log1p(PM2.5) relative to the current value, one LightGBM model per
    (horizon bucket, quantile). Quantiles are invariant to monotone transforms and to adding a
    known offset, so expm1(offset + predicted quantile) is a valid quantile forecast of PM2.5.
    """

    def __init__(self, quantiles=C.QUANTILES, params: dict | None = None):
        self.quantiles = tuple(quantiles)
        self.params = params
        self.models: dict[tuple[int, float], lgb.Booster] = {}
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y) -> Forecaster:
        self.feature_names = list(X.columns)
        offset = np.log1p(X["pm25_now"].to_numpy(float))
        target = np.log1p(np.asarray(y, float)) - offset
        bucket = _bucket_of(X["horizon"].to_numpy())
        for b in range(len(BUCKETS)):
            rows = bucket == b
            if not rows.any():
                raise ValueError(f"no training rows for horizon bucket {BUCKETS[b]}")
            for q in self.quantiles:
                reg = lgb.LGBMRegressor(**_params(q, self.params))
                reg.fit(X[rows], target[rows])
                self.models[(b, q)] = reg.booster_
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names]
        offset = np.log1p(X["pm25_now"].to_numpy(float))
        bucket = _bucket_of(X["horizon"].to_numpy())
        out = np.zeros((len(X), len(self.quantiles)))
        for b in range(len(BUCKETS)):
            rows = bucket == b
            if not rows.any():
                continue
            for j, q in enumerate(self.quantiles):
                out[rows, j] = self.models[(b, q)].predict(X[rows])
        pred = np.clip(np.expm1(out + offset[:, None]), 0, None)
        pred.sort(axis=1)  # independent quantile models can cross; enforce lo <= mid <= hi
        return pd.DataFrame(pred, columns=[f"q{int(q * 100)}" for q in self.quantiles])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for (b, q), booster in self.models.items():
            booster.save_model(str(directory / f"lgbm_b{b}_q{int(q * 100)}.txt"))
        (directory / "features.json").write_text(json.dumps(self.feature_names))

    @classmethod
    def load(cls, directory: Path) -> Forecaster:
        obj = cls()
        obj.feature_names = json.loads((directory / "features.json").read_text())
        for b in range(len(BUCKETS)):
            for q in obj.quantiles:
                path = directory / f"lgbm_b{b}_q{int(q * 100)}.txt"
                obj.models[(b, q)] = lgb.Booster(model_file=str(path))
        return obj


def apply_cqr(pred: pd.DataFrame, horizon: np.ndarray, offsets: dict[str, float]) -> pd.DataFrame:
    """Widen the [q10, q90] band by the conformal offset of each row's horizon bucket."""
    q = np.array([offsets[str(b)] for b in _bucket_of(horizon)])
    out = pred.copy()
    out["q10"] = np.clip(pred["q10"] - q, 0, None)
    out["q90"] = pred["q90"] + q
    return out


def holt_winters_forecast(history: pd.Series, horizon: int = C.HORIZON, window_days: int = 14):
    """Statistical baseline: additive damped-trend Holt-Winters with daily seasonality.

    Fitted on the trailing `window_days` of log1p(PM2.5) at each forecast origin.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # lazy: the app does not need it

    y = np.log1p(history.dropna().iloc[-24 * window_days :].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=24
        ).fit(optimized=True)
    return np.expm1(fit.forecast(horizon)).clip(min=0)
