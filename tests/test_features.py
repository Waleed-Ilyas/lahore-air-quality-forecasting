import numpy as np
import pandas as pd

from aqforecast import features as F


def test_dataset_shape_and_target(synthetic):
    X, meta = F.build_dataset(synthetic, horizons=[1, 24, 72], origin_step=24)
    assert list(X.columns) == F.feature_names() or set(X.columns) == set(F.feature_names())
    assert len(X) == len(meta)
    # y must equal the observed value exactly `horizon` hours after the origin
    lookup = synthetic["pm2_5"]
    expected = lookup.reindex(meta["target_time"]).to_numpy()
    assert np.allclose(meta["y"].to_numpy(), expected)


def test_no_future_leakage(synthetic):
    """Changing PM2.5 *after* the origin must not change any feature for that origin."""
    origin = synthetic.index[24 * 30]
    base, _ = F.build_dataset(synthetic, origins=pd.DatetimeIndex([origin]), require_target=False)
    tampered = synthetic.copy()
    tampered.loc[tampered.index > origin, "pm2_5"] += 1000
    changed, _ = F.build_dataset(tampered, origins=pd.DatetimeIndex([origin]), require_target=False)
    pd.testing.assert_frame_equal(base, changed)


def test_seasonal_naive_uses_same_clock_hour(synthetic):
    X, meta = F.build_dataset(synthetic, horizons=[5, 30], origin_step=37)
    for _, row in meta.iterrows():
        k = -(-row["horizon"] // 24)
        ref_time = row["target_time"] - pd.Timedelta(hours=24 * k)
        assert ref_time <= row["origin"]
        assert row["snaive"] == synthetic["pm2_5"].loc[ref_time]


def test_horizon_and_calendar_features(synthetic):
    X, meta = F.build_dataset(synthetic, horizons=[6], origin_step=50)
    assert (X["horizon"] == 6).all()
    assert X["hour_sin"].between(-1, 1).all() and X["month"].eq(meta["target_time"].dt.month).all()
