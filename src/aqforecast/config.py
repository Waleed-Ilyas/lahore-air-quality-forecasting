"""Project-wide constants: location, API endpoints, paths, modelling settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"  # small files committed to git so the app runs anywhere
FIGURES_DIR = ROOT / "reports" / "figures"

# Lahore city centre. All timestamps are Asia/Karachi local time (UTC+5, no DST), stored tz-naive.
LAT, LON = 31.5497, 74.3436
TIMEZONE = "Asia/Karachi"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# The Open-Meteo air-quality archive starts here.
AQ_START = "2022-08-01"

AQ_VARS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "aerosol_optical_depth", "dust",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "boundary_layer_height",
]

HORIZON = 72  # hours ahead
QUANTILES = (0.1, 0.5, 0.9)  # 80% prediction interval + median
TEST_DAYS = 365  # final hold-out: the most recent full year (contains a full smog season)
GAP_HOURS = HORIZON  # gap between train and test origins so no target window overlaps
SEED = 42
