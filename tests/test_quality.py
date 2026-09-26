import numpy as np
import pandas as pd
import pytest

from aqforecast import quality


def test_check_reports_clean_data(synthetic):
    report = quality.check(synthetic)
    assert report["duplicate_timestamps"] == 0
    assert report["missing_hours"] == 0
    assert report["out_of_range"] == {}


def test_check_flags_out_of_range_and_missing_columns(synthetic):
    bad = synthetic.copy()
    bad.iloc[5, bad.columns.get_loc("relative_humidity_2m")] = 250
    assert quality.check(bad)["out_of_range"] == {"relative_humidity_2m": 1}
    with pytest.raises(ValueError, match="missing columns"):
        quality.check(bad.drop(columns=["pm10"]))


def test_clean_fills_short_gaps_only(synthetic):
    df = synthetic.copy()
    df.iloc[10:13, df.columns.get_loc("pm2_5")] = np.nan  # 3h gap -> filled
    df.iloc[100:120, df.columns.get_loc("pm2_5")] = np.nan  # 20h gap -> stays partly missing
    out = quality.clean(df)
    assert out["pm2_5"].iloc[10:13].notna().all()
    assert out["pm2_5"].iloc[100:120].isna().any()


def test_clean_reindexes_missing_hours_and_drops_duplicates(synthetic):
    df = pd.concat([synthetic.iloc[:50], synthetic.iloc[:1], synthetic.iloc[60:]])
    out = quality.clean(df)
    assert out.index.is_unique
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
