"""US EPA PM2.5 Air Quality Index categories (2024 breakpoints) and alert logic."""
import numpy as np

# (upper bound of PM2.5 in ug/m3, category). 2024 EPA revision of the breakpoints.
BREAKPOINTS = [
    (9.0, "Good"),
    (35.4, "Moderate"),
    (55.4, "Unhealthy for Sensitive Groups"),
    (125.4, "Unhealthy"),
    (225.4, "Very Unhealthy"),
    (float("inf"), "Hazardous"),
]
CATEGORIES = [name for _, name in BREAKPOINTS]
COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]

ALERT_THRESHOLD = 55.5  # "Unhealthy" and above triggers a smog alert
_UPPER = np.array([b for b, _ in BREAKPOINTS[:-1]])


def category_index(pm25) -> np.ndarray:
    """Return the 0-5 category index for PM2.5 values (ug/m3)."""
    values = np.asarray(pm25, dtype=float)
    # AQI concentrations are truncated to 0.1 before lookup, as in the EPA method.
    return np.searchsorted(_UPPER, np.floor(values * 10) / 10, side="left")


def category_name(pm25: float) -> str:
    return CATEGORIES[int(category_index(pm25))]


def category_color(pm25: float) -> str:
    return COLORS[int(category_index(pm25))]


def is_alert(pm25) -> np.ndarray:
    """True where PM2.5 is high enough to trigger a smog alert (Unhealthy or worse)."""
    return np.asarray(pm25, dtype=float) >= ALERT_THRESHOLD
