import numpy as np
import pandas as pd

from aqforecast import evaluate as E
from aqforecast import features as F
from aqforecast import models as M


def _small_xy(df):
    X, meta = F.build_dataset(df, horizons=[1, 5, 10, 20, 30, 60], origin_step=6)
    return X, meta


def test_forecaster_fit_predict_save_load(synthetic, tmp_path):
    X, meta = _small_xy(synthetic)
    f = M.Forecaster(params={"n_estimators": 20, "min_child_samples": 10}).fit(X, meta.y)
    pred = f.predict(X)
    assert list(pred.columns) == ["q10", "q50", "q90"]
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()
    assert (pred >= 0).all().all()
    f.save(tmp_path)
    again = M.Forecaster.load(tmp_path).predict(X)
    pd.testing.assert_frame_equal(pred, again)


def test_forecaster_beats_persistence_on_daily_cycle(synthetic):
    X, meta = _small_xy(synthetic)
    f = M.Forecaster(params={"n_estimators": 60, "min_child_samples": 10}).fit(X, meta.y)
    mae_model = E.point_metrics(meta.y, f.predict(X)["q50"])["mae"]
    mae_persist = E.point_metrics(meta.y, meta.persistence)["mae"]
    assert mae_model < mae_persist


def test_apply_cqr_widens_band():
    pred = pd.DataFrame({"q10": [10.0, 10.0], "q50": [20.0, 20.0], "q90": [30.0, 30.0]})
    offsets = {str(b): 5.0 for b in range(len(M.BUCKETS))}
    out = M.apply_cqr(pred, np.array([1, 72]), offsets)
    assert out["q10"].tolist() == [5.0, 5.0] and out["q90"].tolist() == [35.0, 35.0]


def test_holt_winters_returns_horizon_length(synthetic):
    fc = M.holt_winters_forecast(synthetic["pm2_5"], horizon=72)
    assert len(fc) == 72 and np.isfinite(fc).all()
