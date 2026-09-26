"""Figures (matplotlib, dark/gold theme matching the portfolio site)."""
from __future__ import annotations

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import aqi

BG, PANEL, IVORY, GOLD, BRONZE, CYAN, RED = (
    "#0A0A0C", "#15110F", "#F2EDE4", "#D9B26A", "#8B5E3C", "#4FD1C5", "#ff5a4f",
)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a332d", "axes.labelcolor": IVORY, "text.color": IVORY,
        "xtick.color": IVORY, "ytick.color": IVORY, "grid.color": "#2a2521", "grid.linewidth": 0.8,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "font.family": "DejaVu Sans",
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sql_monthly_and_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EDA aggregations written in SQL (DuckDB) - month x hour climatology and yearly monthly means."""
    tmp = df[["pm2_5"]].reset_index().rename(columns={"time": "ts"})
    con = duckdb.connect()
    con.register("aq", tmp)
    hourly = con.execute(
        """SELECT month(ts) AS month, hour(ts) AS hour, avg(pm2_5) AS pm25
           FROM aq WHERE pm2_5 IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""
    ).df()
    monthly = con.execute(
        """SELECT year(ts) AS year, month(ts) AS month, avg(pm2_5) AS pm25, count(*) AS n
           FROM aq WHERE pm2_5 IS NOT NULL GROUP BY 1, 2 HAVING count(*) > 500 ORDER BY 1, 2"""
    ).df()
    return hourly, monthly


def plot_seasonality(df: pd.DataFrame, path) -> None:
    apply_style()
    hourly, monthly = sql_monthly_and_hourly(df)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    avg = monthly.groupby("month")["pm25"].mean()
    colors = [RED if v >= aqi.ALERT_THRESHOLD else GOLD for v in avg]
    ax[0].bar(MONTHS, avg.reindex(range(1, 13)).values, color=colors)
    ax[0].axhline(aqi.ALERT_THRESHOLD, color=RED, ls="--", lw=1)
    ax[0].set_title("Average PM2.5 by month")
    ax[0].set_ylabel("µg/m³")
    grid = hourly.pivot(index="month", columns="hour", values="pm25")
    im = ax[1].imshow(grid.values, aspect="auto", cmap="YlOrBr", origin="upper")
    ax[1].set_yticks(range(12), MONTHS)
    ax[1].set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax[1].set_title("Diurnal cycle by month (peaks overnight & early morning)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], label="µg/m³")
    _save(fig, path)


def plot_drivers(df: pd.DataFrame, path) -> None:
    """Median PM2.5 by weather bin: shows why smog forms (stagnant, cool, shallow mixing layer)."""
    apply_style()
    con = duckdb.connect()
    con.register("d", df.reset_index())
    specs = [
        ("wind_speed_10m", "Wind speed (km/h)", 8), ("boundary_layer_height", "Boundary-layer height (m)", 8),
        ("temperature_2m", "Temperature (°C)", 8), ("relative_humidity_2m", "Relative humidity (%)", 8),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, (col, label, n) in zip(axes, specs, strict=True):
        q = con.execute(
            f"""WITH b AS (SELECT pm2_5, {col} AS x, ntile({n}) OVER (ORDER BY {col}) AS bin
                           FROM d WHERE pm2_5 IS NOT NULL AND {col} IS NOT NULL)
                SELECT avg(x) AS x, median(pm2_5) AS med, quantile_cont(pm2_5, 0.25) AS q1,
                       quantile_cont(pm2_5, 0.75) AS q3 FROM b GROUP BY bin ORDER BY bin"""
        ).df()
        ax.fill_between(q.x, q.q1, q.q3, color=GOLD, alpha=0.2)
        ax.plot(q.x, q.med, color=GOLD, lw=2.5, marker="o")
        ax.set_xlabel(label)
    axes[0].set_ylabel("PM2.5 µg/m³ (median, IQR band)")
    fig.suptitle("Weather drivers of Lahore PM2.5", fontweight="bold", y=1.0)
    _save(fig, path)


def plot_mae_by_horizon(by_h: dict, path) -> None:
    apply_style()
    names = {"persistence": ("Persistence", BRONZE), "seasonal_naive_3d_mean": ("Seasonal naive", CYAN),
             "holt_winters": ("Holt-Winters", "#B0A99F"), "lgbm": ("LightGBM", GOLD)}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    buckets = list(next(iter(by_h.values())))
    w = 0.2
    for i, (key, (label, color)) in enumerate(names.items()):
        if key in by_h:
            vals = [by_h[key][b]["mae"] for b in buckets]
            bars = ax.bar(np.arange(len(buckets)) + (i - 1.5) * w, vals, w, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, color=IVORY, padding=2)
    ax.set_xticks(range(len(buckets)), buckets)
    ax.set_ylabel("MAE (µg/m³)  ·  lower is better")
    ax.set_title("Forecast error by horizon (hold-out year)")
    ax.legend(ncol=4, loc="upper left")
    _save(fig, path)


def plot_episode(pred: pd.DataFrame, path, days: int = 21) -> None:
    """Worst smog fortnight in the hold-out: actual vs 6h-, 24h- and 72h-ahead forecasts."""
    apply_style()
    a = pred[pred.horizon == 24].set_index("target_time").sort_index()
    roll = a["y"].rolling(24 * days // 3, center=True).mean()  # origins are every 3 h
    end = roll.idxmax()
    lo, hi = end - pd.Timedelta(days=days / 2), end + pd.Timedelta(days=days / 2)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    a24 = a.loc[lo:hi]
    a72 = pred[pred.horizon == 72].set_index("target_time").sort_index().loc[lo:hi]
    ax.fill_between(a24.index, a24.lgbm_q10, a24.lgbm_q90, color=GOLD, alpha=0.18,
                    label="24h-ahead 80% band")
    ax.plot(a72.index, a72.lgbm_q50, color=BRONZE, lw=1.6, label="72h-ahead forecast")
    ax.plot(a24.index, a24.lgbm_q50, color=GOLD, lw=2.2, label="24h-ahead forecast")
    ax.plot(a24.index, a24.y, color=CYAN, lw=1.4, label="Observed")
    ax.axhline(aqi.ALERT_THRESHOLD, color=RED, ls="--", lw=1)
    ax.text(lo, aqi.ALERT_THRESHOLD + 3, "smog alert threshold", color=RED, fontsize=9)
    ax.set_ylabel("PM2.5 µg/m³")
    ax.set_title(f"Smog episode: forecasts vs reality ({lo:%d %b} - {hi:%d %b %Y})")
    ax.legend(ncol=4, loc="upper left")
    _save(fig, path)


def plot_confusion(y, pred, path) -> None:
    apply_style()
    cy, cp = aqi.category_index(y), aqi.category_index(pred)
    n = len(aqi.CATEGORIES)
    cm = np.zeros((n, n))
    for t, p in zip(cy, cp, strict=True):
        cm[t, p] += 1
    norm = cm / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(norm, cmap="YlOrBr", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}%", ha="center", va="center",
                        color=BG if norm[i, j] > 0.5 else IVORY, fontsize=10)
    short = ["Good", "Mod.", "USG", "Unhealthy", "V.Unh.", "Haz."]
    ax.set_xticks(range(n), short)
    ax.set_yticks(range(n), short)
    ax.set_xlabel("Forecast category")
    ax.set_ylabel("Actual category")
    ax.set_title("AQI category: forecast vs actual (row-normalised)")
    ax.grid(False)
    _save(fig, path)


def plot_coverage(pred: pd.DataFrame, path) -> None:
    apply_style()
    g = pred.groupby("horizon")
    raw = g.apply(lambda d: ((d.y >= d.lgbm_q10_raw) & (d.y <= d.lgbm_q90_raw)).mean() * 100,
                  include_groups=False)
    cal = g.apply(lambda d: ((d.y >= d.lgbm_q10) & (d.y <= d.lgbm_q90)).mean() * 100,
                  include_groups=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(raw.index, raw.values, color=BRONZE, lw=2, label="Raw quantile band")
    ax.plot(cal.index, cal.values, color=GOLD, lw=2.5, label="After conformal calibration")
    ax.axhline(80, color=IVORY, ls=":", lw=1)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Actual coverage of the 80% band (%)")
    ax.set_title("Prediction-interval calibration (hold-out year)")
    ax.legend()
    _save(fig, path)


def plot_shap(shap_values: np.ndarray, X: pd.DataFrame, path, top: int = 15) -> None:
    apply_style()
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(np.array(X.columns)[order], imp[order], color=GOLD)
    ax.set_xlabel("mean |SHAP| (change in log1p PM2.5 vs. now)")
    ax.set_title("What drives the 13-24h-ahead forecast")
    _save(fig, path)
