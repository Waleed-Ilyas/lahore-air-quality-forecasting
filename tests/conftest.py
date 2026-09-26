import numpy as np
import pandas as pd
import pytest

from aqforecast import config as C


@pytest.fixture(scope="session")
def synthetic() -> pd.DataFrame:
    """~60 days of synthetic hourly data with a daily cycle, using every column the pipeline needs."""
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", name="time")
    rng = np.random.default_rng(0)
    hour = idx.hour.to_numpy()
    pm = 80 + 40 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 8, len(idx))
    df = pd.DataFrame(index=idx)
    df["pm2_5"] = pm.clip(1)
    for col in C.AQ_VARS:
        if col != "pm2_5":
            df[col] = df["pm2_5"] * rng.uniform(0.5, 1.5)
    df["aerosol_optical_depth"] = df["pm2_5"] / 200  # keep inside the plausible 0-10 range
    df["temperature_2m"] = 15 + 5 * np.sin(2 * np.pi * hour / 24)
    df["relative_humidity_2m"] = 60 + rng.normal(0, 5, len(idx))
    df["dew_point_2m"] = 8.0
    df["precipitation"] = 0.0
    df["surface_pressure"] = 1010.0
    df["cloud_cover"] = 20.0
    df["wind_speed_10m"] = 5 + rng.random(len(idx))
    df["wind_direction_10m"] = 90.0
    df["boundary_layer_height"] = 300 + 200 * np.sin(2 * np.pi * hour / 24)
    return df
