"""Lahore Air Quality Forecast & Smog Alert - Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqforecast import aqi, inference  # noqa: E402
from aqforecast import config as C  # noqa: E402

GOLD, IVORY, BG, PANEL, CYAN = "#D9B26A", "#F2EDE4", "#0A0A0C", "#15110F", "#4FD1C5"
HEALTH_ADVICE = {
    "Good": "Air quality is good. Enjoy outdoor activities.",
    "Moderate": "Acceptable. Unusually sensitive people should reduce prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, older adults and people with heart or lung "
    "disease should limit outdoor exertion.",
    "Unhealthy": "Everyone may feel effects. Wear an N95 mask outdoors and move heavy exercise indoors.",
    "Very Unhealthy": "Health alert. Avoid outdoor activity, keep windows closed, use air purifiers.",
    "Hazardous": "Emergency conditions. Stay indoors; schools and outdoor work should pause.",
}

st.set_page_config(page_title="Lahore Air Quality Forecast", page_icon="🌫️", layout="wide")
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 2rem; max-width: 1200px;}}
    h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}}
    h1 {{font-weight: 400; font-size: 2.6rem;}}
    .banner {{border-radius: 12px; padding: 1rem 1.25rem; margin: .5rem 0 1rem 0;
             border: 1px solid rgba(242,237,228,.12);}}
    .kpi {{background: {PANEL}; border: 1px solid rgba(242,237,228,.08); border-radius: 12px;
          padding: 1rem 1.2rem;}}
    .kpi .label {{font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6;}}
    .kpi .value {{font-size: 2rem; font-family: Georgia, serif; line-height: 1.2;}}
    .kpi .sub {{font-size: .85rem; opacity: .75;}}
    .dot {{display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.4rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def layout(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=IVORY, family="Inter, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(242,237,228,.06)")
    fig.update_yaxes(gridcolor="rgba(242,237,228,.06)")
    return fig


@st.cache_data(ttl=1800, show_spinner="Fetching the latest air-quality data...")
def load_forecast():
    """Live forecast; falls back to the snapshot committed by the daily GitHub Action."""
    try:
        fc, obs, origin = inference.live_forecast()
        return fc, obs, origin, False
    except Exception:  # network / API failure -> stale snapshot
        fc = pd.read_parquet(C.ARTIFACTS_DIR / "latest_forecast.parquet")
        obs = pd.read_parquet(C.ARTIFACTS_DIR / "latest_obs.parquet")
        return fc, obs, pd.Timestamp(fc["origin"].iloc[0]), True


@st.cache_data
def load_json(name: str):
    path = C.ARTIFACTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def kpi(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{dot}{value}</div><div class="sub">{sub}</div></div>')


def forecast_chart(fc: pd.DataFrame, obs: pd.DataFrame, origin: pd.Timestamp) -> go.Figure:
    top = max(150.0, float(fc["q90"].max()) * 1.1, float(obs["pm2_5"].max()) * 1.05)
    fig = go.Figure()
    lower = 0.0
    for (upper, _name), color in zip(aqi.BREAKPOINTS, aqi.COLORS, strict=True):
        if lower >= top:
            break
        fig.add_hrect(y0=lower, y1=min(upper, top), fillcolor=color, opacity=0.07, line_width=0,
                      layer="below")
        lower = upper
    past = obs[obs.index >= origin - pd.Timedelta(hours=48)]
    fig.add_trace(go.Scatter(x=past.index, y=past["pm2_5"], name="Observed (CAMS)",
                             line=dict(color=CYAN, width=2)))
    fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["q90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["q10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(217,178,106,.22)", name="80% prediction band"))
    fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["q50"], name="Forecast (median)",
                             line=dict(color=GOLD, width=3)))
    fig.add_vline(x=origin, line_dash="dot", line_color=IVORY, opacity=0.5)
    fig.add_annotation(x=origin, y=0.02, yref="paper", text="now", showarrow=False, xanchor="right",
                       font=dict(color=IVORY, size=11))
    fig.add_hline(y=aqi.ALERT_THRESHOLD, line_dash="dash", line_color="#ff5a4f", opacity=0.6,
                  annotation_text="smog alert threshold (55.5)", annotation_position="top left",
                  annotation_font_color="#ff5a4f")
    fig.update_yaxes(title="PM2.5 (µg/m³)", range=[0, top])
    return layout(fig)


def render_now(fc, obs, origin, stale):
    now_val = float(obs["pm2_5"].iloc[-1])
    now_cat, now_col = aqi.category_name(now_val), aqi.category_color(now_val)
    s = inference.summarise(fc)
    st.markdown(f"### Right now in Lahore · {origin:%a %d %b, %H:%M}")
    if stale:
        st.warning("Live data source unreachable - showing the most recent saved forecast.")

    if s["alert_hours"]:
        first = pd.Timestamp(s["first_alert"])
        st.markdown(
            f'<div class="banner" style="background:rgba(255,90,79,.12);border-color:#ff5a4f">'
            f"<b>⚠️ Smog alert.</b> Forecast shows <b>{s['alert_hours']} hours</b> of Unhealthy air "
            f"(PM2.5 ≥ 55.5) in the next 72 h, starting <b>{first:%a %H:%M}</b>. "
            f"Peak: <b>{s['peak_pm25']:.0f} µg/m³</b> ({s['peak_category']}) around "
            f"{pd.Timestamp(s['peak_time']):%a %H:%M}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner" style="background:rgba(0,228,0,.08);border-color:#00e400">'
            f"✅ <b>No smog alert.</b> No hour in the next 72 h is forecast to reach Unhealthy levels. "
            f"Peak: {s['peak_pm25']:.0f} µg/m³ ({s['peak_category']}).</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("PM2.5 now", f"{now_val:.0f} µg/m³", now_cat, now_col), unsafe_allow_html=True)
    c2.markdown(kpi("72h peak", f"{s['peak_pm25']:.0f}", f"{s['peak_category']} · "
                    f"{pd.Timestamp(s['peak_time']):%a %H:%M}", aqi.category_color(s["peak_pm25"])),
                unsafe_allow_html=True)
    c3.markdown(kpi("Unhealthy hours", f"{s['alert_hours']} / 72", "forecast PM2.5 ≥ 55.5"),
                unsafe_allow_html=True)
    win = s["next_clean_window"]
    c4.markdown(kpi("Next cleaner air (< 35)", f"{pd.Timestamp(win):%a %H:%M}" if win is not None
                    else "none", "first Moderate-or-better hour"), unsafe_allow_html=True)
    st.info(HEALTH_ADVICE[now_cat], icon="🩺")


def render_forecast(fc, obs, origin):
    st.plotly_chart(forecast_chart(fc, obs, origin), width="stretch")
    daily = fc.assign(day=fc["target_time"].dt.date).groupby("day").agg(
        mean=("q50", "mean"), peak=("q50", "max"), low=("q10", "min"), high=("q90", "max"))
    cols = st.columns(len(daily))
    for col, (day, row) in zip(cols, daily.iterrows(), strict=True):
        col.markdown(kpi(pd.Timestamp(day).strftime("%a %d %b"), f"{row['mean']:.0f} avg",
                         f"peak {row['peak']:.0f} · {aqi.category_name(row['peak'])}",
                         aqi.category_color(row["peak"])), unsafe_allow_html=True)
    st.caption("Median forecast with a calibrated 80% prediction band. Days are partial at the "
               "edges of the 72-hour window.")


def render_trends():
    path = C.ARTIFACTS_DIR / "daily_pm25.parquet"
    if not path.exists():
        st.info("Historical summary not built yet.")
        return
    d = pd.read_parquet(path)
    d.index = pd.to_datetime(d.index)
    st.markdown("#### Daily PM2.5 since Aug 2022")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["mean"], name="Daily mean",
                             line=dict(color="rgba(79,209,197,.55)", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["mean"].rolling(14, center=True, min_periods=5).mean(),
                             name="14-day average", line=dict(color=GOLD, width=3)))
    fig.add_hline(y=aqi.ALERT_THRESHOLD, line_dash="dash", line_color="#ff5a4f", opacity=0.6)
    fig.update_yaxes(title="PM2.5 (µg/m³)")
    st.plotly_chart(layout(fig, 380), width="stretch")

    monthly = d["mean"].resample("MS").mean()
    hm = monthly.to_frame("v").assign(y=monthly.index.year, m=monthly.index.month)
    grid = hm.pivot(index="y", columns="m", values="v")
    fig2 = go.Figure(go.Heatmap(
        z=grid.values, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                          "Nov", "Dec"], y=grid.index.astype(str),
        colorscale=[[0, "#15110F"], [0.5, "#8B5E3C"], [1, "#D9B26A"]], colorbar=dict(title="µg/m³"),
        hovertemplate="%{y} %{x}: %{z:.0f} µg/m³<extra></extra>"))
    fig2.update_yaxes(autorange="reversed")
    st.markdown("#### Monthly average - the smog season is unmistakable")
    st.plotly_chart(layout(fig2, 300), width="stretch")
    share = (d["mean"] >= aqi.ALERT_THRESHOLD).groupby(d.index.month).mean() * 100
    st.caption(f"Share of days with daily-mean PM2.5 ≥ 55.5: "
               f"{share.get(12, 0):.0f}% of Decembers, {share.get(1, 0):.0f}% of Januaries, "
               f"{share.get(7, 0):.0f}% of Julys.")


def render_model():
    m = load_json("metrics.json")
    if not m:
        st.info("Run `python -m aqforecast.train` to generate metrics.")
        return
    h, s = m["holdout"], m["holdout_shared_sample"]
    st.markdown(f"#### Held-out test: the most recent {C.TEST_DAYS} days "
                f"({m['data']['holdout_start'][:10]} → {m['data']['end'][:10]})")
    a, b, c, d = st.columns(4)
    a.markdown(kpi("MAE", f"{h['lgbm']['mae']:.1f} µg/m³",
                   f"seasonal-naive {h['seasonal_naive_3d_mean']['mae']:.1f}"), unsafe_allow_html=True)
    b.markdown(kpi("R²", f"{h['lgbm']['r2']:.2f}", f"seasonal-naive "
                   f"{h['seasonal_naive_3d_mean']['r2']:.2f}"), unsafe_allow_html=True)
    al = m["alerts"]["lgbm"]
    c.markdown(kpi("Alert recall", f"{al['alert_recall'] * 100:.0f}%", "Unhealthy hours caught"),
               unsafe_allow_html=True)
    d.markdown(kpi("Alert precision", f"{al['alert_precision'] * 100:.0f}%", "alerts that were right"),
               unsafe_allow_html=True)

    fig = go.Figure()
    names = {"lgbm": ("LightGBM (this app)", GOLD), "seasonal_naive_3d_mean": ("Seasonal naive", CYAN),
             "persistence": ("Persistence", "#8B5E3C"), "holt_winters": ("Holt-Winters", "#B0A99F")}
    for key, (label, color) in names.items():
        bh = m["by_horizon"].get(key)
        if bh:
            fig.add_trace(go.Bar(x=list(bh), y=[v["mae"] for v in bh.values()], name=label,
                                 marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MAE (µg/m³) - lower is better")
    st.markdown("#### Error by forecast horizon")
    st.plotly_chart(layout(fig, 360), width="stretch")
    st.caption(f"Holt-Winters was evaluated on a {s['origins']}-origin random sample of the "
               f"hold-out year (it is slow to refit); all other bars use every hold-out hour.")
    iv = m["interval"]["cqr_80pct_band"]
    st.write(f"**80% band:** covers {iv['coverage_pct']:.1f}% of actual hours "
             f"(raw model band: {m['interval']['raw_80pct_band']['coverage_pct']:.1f}%, calibrated with "
             f"conformal prediction).")
    live = load_json("live_accuracy.json")
    if live and live.get("n"):
        st.write(f"**Live tracking:** {live['n']:,} forecast-hours issued since "
                 f"{live['since'][:10]} have been checked against observations - MAE "
                 f"{live['mae_all']:.1f} µg/m³.")


def render_about():
    st.markdown(
        """
**What this is.** A production-style forecasting system for Lahore's PM2.5, built for citizens,
schools and businesses who need to plan around smog days.

**Data.** Hourly PM2.5, PM10, NO₂, O₃, CO, SO₂, dust and aerosol depth plus weather (temperature,
humidity, wind, boundary-layer height, pressure, precipitation) from the free
[Open-Meteo](https://open-meteo.com) APIs (CC BY 4.0), Aug 2022 → today. **Important:** the
air-quality values are CAMS atmospheric-model output for a ~10 km grid cell, *not* readings from a
ground sensor, so they can differ from the US Embassy monitor or a street-level sensor.

**Model.** 18 LightGBM models (6 horizon buckets × 3 quantiles) predict the change in PM2.5 from now,
using lags, rolling statistics, current pollutants and the weather forecast for the target hour.
The 80% band is calibrated with conformalised quantile regression on out-of-fold residuals.

**Alerts.** "Unhealthy" and above (PM2.5 ≥ 55.5 µg/m³, US EPA 2024 breakpoints).
        """
    )


st.title("Lahore Air Quality Forecast")
st.caption("72-hour PM2.5 forecast & smog alerts · data: Open-Meteo / CAMS · model: LightGBM "
           "quantile forecaster")
forecast, observed, origin_ts, is_stale = load_forecast()
render_now(forecast, observed, origin_ts, is_stale)
tab1, tab2, tab3, tab4 = st.tabs(["72-hour forecast", "Historical trends", "Model performance", "About"])
with tab1:
    render_forecast(forecast, observed, origin_ts)
with tab2:
    render_trends()
with tab3:
    render_model()
with tab4:
    render_about()
