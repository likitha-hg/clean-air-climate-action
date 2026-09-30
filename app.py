from pathlib import Path
from tempfile import NamedTemporaryFile
from html import escape

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.gemini.analyze_citizen_report import analyze_citizen_report


# ============================================================
# PAGE CONFIG
# ============================================================

PROJECT_NAME = "Clean Air & Climate Action"
PROJECT_SUBTITLE = "AI-powered environmental intelligence"

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

AIR_QUALITY_FILE = PROJECT_ROOT / "data/processed/air_quality_clean.csv"
FORECAST_FILE = PROJECT_ROOT / "data/processed/pm25_forecast_dataset.csv"
HOTSPOT_FILE = PROJECT_ROOT / "data/processed/hotspots_latest.csv"
ALERT_FILE = PROJECT_ROOT / "data/processed/authority_alerts_final.csv"
MODEL_FILE = PROJECT_ROOT / "models/pm25_xgb_model.joblib"
FEDERATED_MODEL_FILE = PROJECT_ROOT / "models/federated_pm25_model.joblib"
FEDERATED_METRICS_FILE = PROJECT_ROOT / "data/processed/federated_training_metrics.csv"
FEDERATED_CLIENT_METRICS_FILE = PROJECT_ROOT / "data/processed/federated_client_metrics.csv"
FEDERATED_REGIONAL_METRICS_FILE = PROJECT_ROOT / "data/processed/federated_regional_evaluation.csv"


# ============================================================
# GLOBAL UI
# ============================================================

st.html(
    """
    <style>
    :root {
        --ink: #10211d;
        --muted: #6c7b77;
        --bg: #f5f8f6;
        --white: #ffffff;
        --line: rgba(16,33,29,.08);
        --teal: #0f766e;
        --teal-dark: #0a5c56;
        --teal-soft: #dff4ef;
        --green: #2f855a;
        --green-soft: #e7f6ec;
        --amber: #c0841a;
        --amber-soft: #fff4da;
        --red: #c2413b;
        --red-soft: #fde8e7;
        --critical: #8f2020;
        --critical-soft: #fbe3e3;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(15,118,110,.08), transparent 24%),
            radial-gradient(circle at 92% 8%, rgba(47,133,90,.06), transparent 20%),
            var(--bg);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 5.75rem !important;
        padding-bottom: 4rem;
    }

    /* Keep the application content safely below Streamlit's fixed top chrome. */
    [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }

    [data-testid="stHeader"] {
        z-index: 1000;
    }

    header[data-testid="stHeader"] {
        background: rgba(245,248,246,.78);
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(16,33,29,.05);
    }

    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none;
    }

    .top-navigation {
        width: 100%;
        padding: 8px 0 12px 0;
        margin: 4px 0 18px 0;
        border-bottom: 1px solid rgba(16,33,29,.07);
    }

    .nav-spacer {
        min-height: 1px;
    }

    [data-testid="stPageLink"] {
        margin-top: 0 !important;
    }

    [data-testid="stPageLink"] a {
        border-radius: 12px !important;
        padding: 7px 10px !important;
        color: #53635f !important;
        text-decoration: none !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        transition: background .2s ease, color .2s ease, transform .2s ease !important;
    }

    [data-testid="stPageLink"] a:hover {
        background: rgba(15,118,110,.08) !important;
        color: #0a5c56 !important;
        transform: translateY(-1px);
    }

    /* Extra clearance keeps the custom header below Streamlit's fixed chrome on desktop and during browser zoom. */
    .app-top-safe {
        position: relative;
        z-index: 2;
        padding-top: 4px;
    }

    .brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin: 10px 0 14px;
        animation: rise .55s ease both;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        flex: 0 0 46px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        color: #fff;
        background: linear-gradient(145deg, #0a5c56 0%, #0f766e 52%, #2f855a 100%);
        box-shadow:
            0 12px 28px rgba(15,118,110,.22),
            inset 0 1px 0 rgba(255,255,255,.20);
        position: relative;
        overflow: hidden;
    }

    .brand-mark::after {
        content: "";
        position: absolute;
        width: 28px;
        height: 28px;
        top: -10px;
        right: -10px;
        border-radius: 50%;
        background: rgba(255,255,255,.10);
    }

    .brand-name {
        color: var(--ink);
        font-size: 16px;
        font-weight: 800;
        letter-spacing: -.02em;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 10px;
        margin-top: 3px;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .topnav-caption {
        margin: 0 0 6px;
        color: var(--muted);
        font-size: 8px;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
        text-align: right;
    }

    .topnav-divider {
        height: 1px;
        margin: 0 0 10px;
        background: linear-gradient(90deg, transparent, rgba(16,33,29,.08));
    }

    [data-testid="stPageLink"] {
        margin: 0 !important;
    }

    [data-testid="stPageLink"] a {
        border-radius: 12px !important;
        padding: 7px 8px !important;
        min-height: 34px !important;
        transition: transform .18s ease, background .18s ease, box-shadow .18s ease;
    }

    [data-testid="stPageLink"] a:hover {
        transform: translateY(-2px);
        background: rgba(223,244,239,.72);
        box-shadow: 0 8px 20px rgba(16,33,29,.06);
    }

    [data-testid="stPageLink"] a p {
        font-size: 10px !important;
        font-weight: 800 !important;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        animation: pulse 2.2s infinite;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 42px 42px;
        margin-bottom: 24px;
        color: #fff;
        background: linear-gradient(135deg,#0d302b 0%,#0f766e 56%,#25785c 100%);
        box-shadow: 0 24px 70px rgba(15,118,110,.18);
        animation: rise .7s .05s ease both;
    }

    .hero::before,
    .hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,.06);
        pointer-events: none;
    }

    .hero::before {
        width: 360px;
        height: 360px;
        right: -130px;
        top: -180px;
        animation: drift 10s ease-in-out infinite;
    }

    .hero::after {
        width: 220px;
        height: 220px;
        right: 140px;
        bottom: -170px;
        animation: drift 13s ease-in-out infinite reverse;
    }

    .eyebrow {
        position: relative;
        z-index: 1;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .16em;
        opacity: .75;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        margin: 10px 0 0;
        max-width: 760px;
        font-size: clamp(34px,4.5vw,60px);
        line-height: .98;
        letter-spacing: -.045em;
        font-weight: 800;
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 730px;
        margin-top: 17px;
        color: rgba(255,255,255,.78);
        font-size: 15px;
        line-height: 1.7;
    }

    .hero-tags {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 22px;
    }

    .hero-tag {
        padding: 7px 10px;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        color: rgba(255,255,255,.85);
        font-size: 10px;
        letter-spacing: .04em;
    }

    .data-strip {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: wrap;
        padding: 11px 14px;
        margin-bottom: 20px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(255,255,255,.7);
        color: var(--muted);
        font-size: 10px;
    }

    .section {
        margin: 34px 0 14px;
        animation: rise .55s ease both;
    }

    .section-title {
        margin: 0;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -.025em;
        color: var(--ink);
    }

    .section-description {
        margin-top: 4px;
        font-size: 12px;
        line-height: 1.55;
        color: var(--muted);
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(4,minmax(0,1fr));
        gap: 12px;
    }

    .metric {
        position: relative;
        overflow: hidden;
        padding: 20px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.82);
        box-shadow: 0 12px 40px rgba(16,33,29,.05);
        transition: transform .22s ease, box-shadow .22s ease;
        animation: rise .6s ease both;
    }

    .metric:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 50px rgba(16,33,29,.1);
    }

    .metric::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg,#0f766e,#2f855a);
    }

    .metric-label {
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .09em;
    }

    .metric-value {
        margin-top: 8px;
        color: var(--ink);
        font-size: 30px;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -.04em;
    }

    .metric-support {
        margin-top: 8px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.5;
    }

    .panel {
        padding: 18px;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: rgba(255,255,255,.8);
        box-shadow: 0 12px 40px rgba(16,33,29,.05);
    }

    .panel-title {
        font-size: 13px;
        font-weight: 800;
        color: var(--ink);
    }

    .panel-subtitle {
        margin-top: 4px;
        font-size: 11px;
        color: var(--muted);
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3,minmax(0,1fr));
        gap: 12px;
    }

    .feature {
        padding: 20px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(145deg,rgba(255,255,255,.92),rgba(246,250,248,.72));
        box-shadow: 0 12px 40px rgba(16,33,29,.05);
        transition: transform .22s ease, box-shadow .22s ease;
        animation: rise .6s ease both;
    }

    .feature:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 50px rgba(16,33,29,.09);
    }

    .feature-icon {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        margin-bottom: 14px;
        background: var(--teal-soft);
        color: var(--teal-dark);
        font-size: 12px;
        font-weight: 800;
    }

    .feature-title {
        color: var(--ink);
        font-size: 14px;
        font-weight: 800;
    }

    .feature-copy {
        margin-top: 7px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.65;
    }

    .alerts {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .alert {
        padding: 14px;
        border: 1px solid var(--line);
        border-radius: 17px;
        background: rgba(255,255,255,.88);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .alert:hover {
        transform: translateX(4px);
        box-shadow: 0 12px 30px rgba(16,33,29,.07);
    }

    .alert-top {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
    }

    .alert-station {
        color: var(--ink);
        font-size: 13px;
        font-weight: 800;
    }

    .badge {
        display: inline-flex;
        padding: 5px 8px;
        border-radius: 999px;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: .06em;
    }

    .badge-critical { color: var(--critical); background: var(--critical-soft); }
    .badge-high { color: var(--red); background: var(--red-soft); }
    .badge-moderate { color: var(--amber); background: var(--amber-soft); }
    .badge-low { color: var(--green); background: var(--green-soft); }

    .alert-grid {
        display: grid;
        grid-template-columns: repeat(3,minmax(0,1fr));
        gap: 8px;
        margin-top: 11px;
    }

    .alert-box {
        padding: 8px;
        border-radius: 11px;
        background: #f6f9f7;
    }

    .alert-label {
        color: var(--muted);
        font-size: 8px;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .alert-value {
        margin-top: 4px;
        color: var(--ink);
        font-size: 14px;
        font-weight: 800;
    }

    .alert-reason {
        margin-top: 10px;
        color: #5a6965;
        font-size: 11px;
        line-height: 1.55;
    }

    .ai-box {
        padding: 20px;
        border: 1px solid rgba(15,118,110,.12);
        border-radius: 24px;
        background: radial-gradient(circle at 100% 0%,rgba(15,118,110,.1),transparent 32%),#fff;
        box-shadow: 0 18px 48px rgba(16,33,29,.08);
    }

    .ai-label {
        color: var(--teal-dark);
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .1em;
    }

    .ai-title {
        margin-top: 5px;
        color: var(--ink);
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -.035em;
    }

    .node-grid {
        display: grid;
        grid-template-columns: repeat(5,minmax(0,1fr));
        gap: 10px;
    }

    .node {
        padding: 16px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 12px 35px rgba(16,33,29,.05);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .node:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(16,33,29,.08);
    }

    .node-line {
        height: 2px;
        margin-bottom: 12px;
        background: linear-gradient(90deg,#0f766e,#2f855a);
    }

    .node-name {
        color: var(--ink);
        font-size: 13px;
        font-weight: 800;
    }

    .node-state {
        margin-top: 7px;
        color: var(--green);
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .footer {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 10px;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes drift {
        0%,100% { transform: translate3d(0,0,0); }
        50% { transform: translate3d(-18px,16px,0); }
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(47,133,90,.45); }
        70% { box-shadow: 0 0 0 8px rgba(47,133,90,0); }
        100% { box-shadow: 0 0 0 0 rgba(47,133,90,0); }
    }

    @media (max-width: 1100px) {
        .metrics { grid-template-columns: repeat(2,minmax(0,1fr)); }
        .feature-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
        .node-grid { grid-template-columns: repeat(3,minmax(0,1fr)); }
    }

    @media (max-width: 700px) {
        .hero { padding: 30px 22px; border-radius: 22px; }
        .metrics,.feature-grid,.node-grid { grid-template-columns: 1fr; }
        .status-row { display: none; }
    }

    @media (prefers-reduced-motion: reduce) {
        *,*::before,*::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
    }
    </style>
    """
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data

def load_data():
    air_quality = pd.read_csv(AIR_QUALITY_FILE, parse_dates=["timestamp"])
    forecast = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    hotspots = pd.read_csv(HOTSPOT_FILE)
    alerts = pd.read_csv(ALERT_FILE)
    return air_quality, forecast, hotspots, alerts


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"], bundle["features"]


@st.cache_resource
def load_federated_model():
    return joblib.load(FEDERATED_MODEL_FILE)


@st.cache_data
def load_federated_metrics():
    global_metrics = pd.read_csv(FEDERATED_METRICS_FILE)
    client_metrics = pd.read_csv(FEDERATED_CLIENT_METRICS_FILE)
    regional_metrics = pd.read_csv(FEDERATED_REGIONAL_METRICS_FILE)
    return global_metrics, client_metrics, regional_metrics


try:
    air_quality, forecast_data, hotspots, alerts = load_data()
    federated_bundle = load_federated_model()
    (
        federated_global_metrics,
        federated_client_metrics,
        federated_regional_metrics,
    ) = load_federated_metrics()
except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()


# ============================================================
# COMMON STATE
# ============================================================

latest_timestamp = air_quality["timestamp"].max()
latest_air = air_quality[air_quality["timestamp"] == latest_timestamp].copy()

critical_alerts = alerts[alerts["alert_priority"] == "CRITICAL_REVIEW"]
high_alerts = alerts[alerts["alert_priority"] == "HIGH"]
moderate_alerts = alerts[alerts["alert_priority"] == "MODERATE"]
low_alerts = alerts[alerts["alert_priority"] == "LOW"]


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    return escape(str(value))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def format_date(value):
    try:
        return pd.Timestamp(value).strftime("%d %b %Y · %H:%M UTC")
    except Exception:
        return str(value)


def priority_class(priority):
    return {
        "CRITICAL_REVIEW": "critical",
        "HIGH": "high",
        "MODERATE": "moderate",
        "LOW": "low",
    }.get(str(priority), "low")


def render_brand_bar():
    # Kept as a reusable helper for any future page-specific layout.
    st.html(
        f"""
        <div class="brand" style="margin-bottom:0;">
            <div class="brand-left">
                <div class="brand-mark">
                    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <path d="M22.6 7.1C18.2 7.5 13.8 9.1 11 12.2C8.9 14.5 8.3 17.4 9.9 19.7C11.7 22.3 15.1 22.7 17.8 20.9C20.9 18.9 22.4 14.2 22.6 7.1Z" fill="white" fill-opacity="0.96"/>
                        <path d="M9.1 24.8C12.2 20.4 15.8 17.1 21.8 13.1" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
                        <path d="M6 11.4C8.3 9.6 10.4 9 13 9.2" stroke="white" stroke-opacity="0.62" stroke-width="1.6" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div class="brand-name">{safe_text(PROJECT_NAME)}</div>
                    <div class="brand-subtitle">{safe_text(PROJECT_SUBTITLE)}</div>
                </div>
            </div>
        </div>
        """
    )


def render_data_strip():
    st.html(
        f"""
        <div class="data-strip">
            <span>
                <strong style="color:#10211d;">Synchronized dataset</strong>
                · Latest observation: {safe_text(format_date(latest_timestamp))}
            </span>
            <span>
                44 monitoring stations · 33 federated model features
            </span>
        </div>
        """
    )


def render_hero(eyebrow, title, copy, tags):
    tag_html = "".join(
        f'<span class="hero-tag">{safe_text(tag)}</span>'
        for tag in tags
    )

    st.html(
        f"""
        <div class="hero">
            <div class="eyebrow">{safe_text(eyebrow)}</div>
            <h1 class="hero-title">{title}</h1>
            <div class="hero-copy">{safe_text(copy)}</div>
            <div class="hero-tags">{tag_html}</div>
        </div>
        """
    )


def render_section(title, description=""):
    description_html = (
        f'<div class="section-description">{safe_text(description)}</div>'
        if description
        else ""
    )
    st.html(
        f"""
        <div class="section">
            <h2 class="section-title">{safe_text(title)}</h2>
            {description_html}
        </div>
        """
    )


def render_metrics(cards):
    html = '<div class="metrics">'
    for card in cards:
        html += f"""
        <div class="metric">
            <div class="metric-label">{safe_text(card['label'])}</div>
            <div class="metric-value">{safe_text(card['value'])}</div>
            <div class="metric-support">{safe_text(card['support'])}</div>
        </div>
        """
    html += "</div>"
    st.html(html)


def render_features(items):
    html = '<div class="feature-grid">'
    for item in items:
        html += f"""
        <div class="feature">
            <div class="feature-icon">{safe_text(item['icon'])}</div>
            <div class="feature-title">{safe_text(item['title'])}</div>
            <div class="feature-copy">{safe_text(item['copy'])}</div>
        </div>
        """
    html += "</div>"
    st.html(html)


def render_alert_cards(frame, limit=6):
    if frame.empty:
        st.html(
            """
            <div class="panel">
                <div class="panel-title">No alerts in this view</div>
                <div class="panel-subtitle">No current authority records match this section.</div>
            </div>
            """
        )
        return

    html = '<div class="alerts">'
    for _, row in frame.head(limit).iterrows():
        priority = str(row.get("alert_priority", "LOW"))
        css = priority_class(priority)
        current = float(row.get("pm25", 0) or 0)
        predicted = float(row.get("predicted_pm25_3h", 0) or 0)
        hotspot = float(row.get("hotspot_score", 0) or 0)
        reason = safe_text(row.get("alert_reason", "No reason provided."))
        station = safe_text(row.get("station", "Unknown station"))

        html += f"""
        <div class="alert">
            <div class="alert-top">
                <div class="alert-station">{station}</div>
                <div class="badge badge-{css}">{safe_text(priority)}</div>
            </div>
            <div class="alert-grid">
                <div class="alert-box"><div class="alert-label">PM2.5</div><div class="alert-value">{current:.1f}</div></div>
                <div class="alert-box"><div class="alert-label">3H FORECAST</div><div class="alert-value">{predicted:.1f}</div></div>
                <div class="alert-box"><div class="alert-label">HOTSPOT</div><div class="alert-value">{hotspot:.2f}</div></div>
            </div>
            <div class="alert-reason">{reason}</div>
        </div>
        """
    html += "</div>"
    st.html(html)


def styled_plot(fig, height=460):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#31423e"},
        margin={"l": 8, "r": 8, "t": 45, "b": 8},
        height=height,
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(16,33,29,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(16,33,29,0.06)", zeroline=False)
    return fig


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview():
    render_data_strip()
    render_hero(
        "CLEAN AIR & CLIMATE ACTION",
        "See the signal.<br>Understand the pattern.",
        "An AI-powered platform for air-quality monitoring, PM2.5 forecasting, pollution hotspot detection, satellite fire intelligence, citizen reporting, and coordinated climate action.",
        [
            "AIR QUALITY",
            "PM2.5 FORECASTING",
            "POLLUTION HOTSPOTS",
            "SATELLITE FIRE DETECTION",
            "GEMINI AI",
            "CLIMATE ACTION",
        ],
    )

    render_section(
        "National air quality",
        "A compact view of the latest synchronized environmental picture.",
    )
    render_metrics(
        [
            {"label": "Monitoring stations", "value": str(air_quality["station_id"].nunique()), "support": "Current monitoring network"},
            {"label": "Latest PM2.5 points", "value": str(latest_air["station_id"].nunique()), "support": "Stations in the latest snapshot"},
            {"label": "Moderate alerts", "value": str(len(moderate_alerts)), "support": "Authority attention recommended"},
            {"label": "High alerts", "value": str(len(high_alerts)), "support": "Escalated review currently required"},
        ]
    )

    render_section(
        "Monitoring network",
        "Current PM2.5 intensity across the available station network.",
    )

    map_df = latest_air[["station", "latitude", "longitude", "pm25", "pm10", "no2"]].dropna(
        subset=["latitude", "longitude", "pm25"]
    )

    st.html(
        """
        <div class="panel">
            <div class="panel-title">Air quality monitoring map</div>
            <div class="panel-subtitle">World view of current India monitoring locations. Point size and intensity represent PM2.5.</div>
        </div>
        """
    )
    fig = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        size="pm25",
        color="pm25",
        hover_name="station",
        hover_data={
            "pm25": ":.1f",
            "pm10": ":.1f",
            "no2": ":.1f",
            "latitude": False,
            "longitude": False,
        },
        scope="world",
        projection="natural earth",
        color_continuous_scale=[
            [0.00, "#2f855a"],
            [0.25, "#7fb069"],
            [0.50, "#f2c14e"],
            [0.75, "#e67e22"],
            [1.00, "#c2413b"],
        ],
        range_color=(0, max(30, float(map_df["pm25"].quantile(0.95)) if not map_df.empty else 30)),
    )
    fig.update_geos(
        center={"lat": 22.5, "lon": 79.0},
        projection_scale=4.5,
        showland=True,
        landcolor="#edf2ef",
        showcountries=True,
        countrycolor="#ccd6d2",
        showcoastlines=True,
        coastlinecolor="#d8e0dd",
    )
    fig.update_traces(
        marker={
            "line": {"width": 0.8, "color": "rgba(255,255,255,0.85)"},
            "colorbar": {
                "title": {
                    "text": "PM2.5<br>µg/m³",
                    "font": {"size": 11, "color": "#31423e"},
                },
                "tickmode": "array",
                "tickvals": [10, 15, 20, 25, 30],
                "ticktext": ["10", "15", "20", "25", "30+"],
                "ticks": "outside",
                "ticklen": 4,
                "tickfont": {"size": 10, "color": "#4d5f5a"},
                "thickness": 14,
                "len": 0.72,
                "x": 1.02,
                "y": 0.5,
                "bgcolor": "rgba(255,255,255,0.78)",
                "borderwidth": 0,
            },
        }
    )
    styled_plot(fig, 450)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.html(
        """
        <div class="panel" style="margin-top:8px; margin-bottom:10px;">
            <div class="panel-title">Priority alert feed</div>
            <div class="panel-subtitle">Current authority signals requiring attention.</div>
        </div>
        """
    )
    order = {"CRITICAL_REVIEW": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    feed = alerts.copy()
    feed["_sort"] = feed["alert_priority"].map(order).fillna(9)
    feed = feed.sort_values(["_sort", "recent_event_score", "hotspot_score"], ascending=[True, False, False])
    render_alert_cards(feed.drop(columns="_sort"), limit=6)

    render_section(
        "Current PM2.5 pattern",
        "The highest PM2.5 values in the latest synchronized snapshot.",
    )
    top_locations = latest_air[["station", "pm25", "pm10", "no2"]].sort_values("pm25", ascending=False).head(10)
    display_df = top_locations.rename(columns={"station": "Station", "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂"}).reset_index(drop=True)
    st.dataframe(display_df, width="stretch", hide_index=True)

    render_section("What the platform connects", "Different signals are combined into one simple environmental intelligence workflow.")
    render_features(
        [
            {"icon": "01", "title": "Observe", "copy": "Ground-level pollutant measurements show what is happening at specific monitoring locations."},
            {"icon": "02", "title": "Predict", "copy": "A machine-learning model estimates PM2.5 approximately three hours ahead."},
            {"icon": "03", "title": "Detect", "copy": "Hotspot scoring highlights locations with unusual or elevated pollution signals."},
            {"icon": "04", "title": "Connect", "copy": "Satellite fire observations add spatial and temporal context to selected pollution events."},
            {"icon": "05", "title": "Understand", "copy": "Gemini multimodal AI interprets citizen-submitted images and descriptions."},
            {"icon": "06", "title": "Act", "copy": "Authority alert context helps turn environmental signals into review and response."},
        ]
    )

    st.html(
        f"""
        <div class="footer">
            <span>{safe_text(PROJECT_NAME)}</span>
            <span>Observation → Prediction → Detection → Action</span>
        </div>
        """
    )


# ============================================================
# PAGE: AIR QUALITY
# ============================================================

def page_air_quality():
    render_data_strip()
    render_hero(
        "AIR QUALITY MONITORING",
        "Understand today's air.",
        "Explore station-level conditions and a three-hour PM2.5 forecast generated by the current machine-learning model.",
        ["PM2.5", "3-HOUR FORECAST", "MACHINE LEARNING"],
    )

    station_lookup = forecast_data[["station_id", "station"]].drop_duplicates().sort_values("station")
    selected_station = st.selectbox("Monitoring station", station_lookup["station"].tolist())
    selected_station_id = station_lookup.loc[station_lookup["station"] == selected_station, "station_id"].iloc[0]

    station_history = forecast_data[forecast_data["station_id"] == selected_station_id].sort_values("timestamp").copy()

    try:
        model, model_features = load_model()
    except Exception as exc:
        st.error(f"Unable to load forecasting model: {exc}")
        return

    latest_row = station_history.dropna(subset=model_features).sort_values("timestamp").tail(1)
    if latest_row.empty:
        st.warning("This station does not have a complete model input row.")
        return

    latest_row = latest_row.iloc[0]
    prediction = float(model.predict(pd.DataFrame([latest_row[model_features]]))[0])
    prediction = max(0.0, prediction)
    current_pm25 = float(latest_row["pm25"])
    forecast_time = latest_row["timestamp"] + pd.Timedelta(hours=3)
    delta = prediction - current_pm25

    render_section("Station conditions", "Current PM2.5 compared with the next three-hour estimate.")
    render_metrics(
        [
            {"label": "Current PM2.5", "value": f"{current_pm25:.1f}", "support": "µg/m³ at latest station reading"},
            {"label": "Predicted PM2.5", "value": f"{prediction:.1f}", "support": "Estimated approximately three hours ahead"},
            {"label": "Change", "value": f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}", "support": "Potential rise" if delta > 0 else "Potential decrease" if delta < 0 else "Relatively stable"},
            {"label": "Forecast horizon", "value": "3H", "support": format_date(forecast_time)},
        ]
    )

    render_section("Recent station trend", "The latest 48 available PM2.5 observations.")
    chart_data = station_history[["timestamp", "pm25"]].tail(48)
    fig = px.line(chart_data, x="timestamp", y="pm25", markers=True, labels={"timestamp": "", "pm25": "PM2.5"})
    fig.update_traces(line={"width": 3}, marker={"size": 6})
    styled_plot(fig, 450)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    render_section("Forecast context", "The model supports decisions but does not guarantee future conditions.")
    context_row = alerts[alerts["station"] == selected_station]
    if not context_row.empty:
        context_row = context_row.iloc[0]
        st.html(
            f"""
            <div class="ai-box">
                <div class="ai-label">CURRENT AUTHORITY CONTEXT</div>
                <div class="ai-title">{safe_text(context_row['alert_priority'])}</div>
                <div style="margin-top:10px;color:#5e6c68;font-size:12px;line-height:1.7;">
                    Forecast risk: <strong style="color:#10211d;">{safe_text(context_row['forecast_risk'])}</strong><br>
                    {safe_text(context_row['alert_reason'])}
                </div>
            </div>
            """
        )


# ============================================================
# PAGE: POLLUTION EVENTS
# ============================================================

def page_events():
    render_data_strip()
    render_hero(
        "POLLUTION EVENTS",
        "Trace the signal. Understand the change.",
        "Combine air-quality observations, hotspot scores, wind context and satellite fire observations to identify events that may need further investigation.",
        ["HOTSPOT DETECTION", "SATELLITE FIRE", "EVENT INTELLIGENCE"],
    )

    linked_stations = int((alerts["recent_event_count"] > 0).sum())
    event_records = int(alerts["recent_event_count"].sum())

    render_section("Event signals", "Current hotspot and satellite-linked information.")
    render_metrics(
        [
            {"label": "High hotspots", "value": str(int((hotspots["risk_level"] == "HIGH").sum())), "support": "Locations with highest current signal"},
            {"label": "Moderate hotspots", "value": str(int((hotspots["risk_level"] == "MODERATE").sum())), "support": "Locations with elevated signals"},
            {"label": "Linked stations", "value": str(linked_stations), "support": "Stations with recent satellite-linked evidence"},
            {"label": "Linked observations", "value": str(event_records), "support": "Recent event links across stations"},
        ]
    )

    render_section("Hotspot landscape", "India monitoring locations ranked by their current hotspot score in a world-view map.")
    hotspot_map = hotspots[["station", "latitude", "longitude", "pm25", "hotspot_score", "risk_level"]].dropna(
        subset=["latitude", "longitude", "hotspot_score"]
    )


    fig = px.scatter_geo(
        hotspot_map,
        lat="latitude",
        lon="longitude",
        size="hotspot_score",
        color="pm25",
        hover_name="station",
        hover_data={
            "pm25": ":.1f",
            "hotspot_score": ":.2f",
            "risk_level": True,
            "latitude": False,
            "longitude": False,
        },
        scope="world",
        projection="natural earth",
        color_continuous_scale=[
            [0.00, "#2f855a"],
            [0.25, "#7fb069"],
            [0.50, "#f2c14e"],
            [0.75, "#e67e22"],
            [1.00, "#c2413b"],
        ],
        range_color=(
            0,
            max(
                30,
                float(hotspot_map["pm25"].quantile(0.95))
                if not hotspot_map.empty
                else 30,
            ),
        ),
    )
    fig.update_geos(
        center={"lat": 22.5, "lon": 79.0},
        projection_scale=4.5,
        showland=True,
        landcolor="#edf2ef",
        showcountries=True,
        countrycolor="#ccd6d2",
        showcoastlines=True,
        coastlinecolor="#d8e0dd",
    )
    fig.update_traces(
        marker={
            "line": {"width": 0.8, "color": "rgba(255,255,255,0.85)"},
            "colorbar": {
                "title": {
                    "text": "PM2.5<br>µg/m³",
                    "font": {"size": 11, "color": "#31423e"},
                },
                "tickmode": "array",
                "tickvals": [10, 15, 20, 25, 30],
                "ticktext": ["10", "15", "20", "25", "30+"],
                "ticks": "outside",
                "ticklen": 4,
                "tickfont": {"size": 10, "color": "#4d5f5a"},
                "thickness": 14,
                "len": 0.72,
                "x": 1.02,
                "y": 0.5,
                "bgcolor": "rgba(255,255,255,0.78)",
                "borderwidth": 0,
            },
        }
    )
    styled_plot(fig, 450)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.html(
        """
        <div class="panel" style="margin-top:8px; margin-bottom:10px;">
            <div class="panel-title">Recent event evidence</div>
            <div class="panel-subtitle">Stations with satellite-linked context in the current alert window.</div>
        </div>
        """
    )
    event_feed = alerts[alerts["recent_event_count"] > 0].sort_values(
        ["recent_event_score", "hotspot_score"], ascending=[False, False]
    )
    render_alert_cards(event_feed, limit=6)

    render_section("Event investigation", "Inspect the evidence behind an individual station signal.")
    station_options = alerts["station"].dropna().sort_values().tolist()
    event_station = st.selectbox("Station", station_options, key="event_station")
    event_row = alerts[alerts["station"] == event_station]

    if not event_row.empty:
        row = event_row.iloc[0]
        source_context = str(row.get("recent_source_context", "No recent source context"))
        priority = str(row.get("alert_priority", "LOW"))
        count = int(row.get("recent_event_count", 0))
        score = float(row.get("recent_event_score", 0) or 0)
        reason = str(row.get("alert_reason", "No assessment available."))
        action = str(row.get("recommended_action", "Review current environmental conditions."))

        st.html(
            f"""
            <div class="ai-box">
                <div class="ai-label">EVIDENCE CHAIN</div>
                <div class="ai-title">{safe_text(source_context)}</div>
                <div class="feature-grid" style="margin-top:18px;">
                    <div class="feature"><div class="feature-title">Authority priority</div><div class="feature-copy">{safe_text(priority)}</div></div>
                    <div class="feature"><div class="feature-title">Linked events</div><div class="feature-copy">{count}</div></div>
                    <div class="feature"><div class="feature-title">Event score</div><div class="feature-copy">{score:.2f}</div></div>
                </div>
                <div style="margin-top:18px;color:#596863;font-size:12px;line-height:1.7;">
                    <strong style="color:#10211d;">Assessment:</strong> {safe_text(reason)}<br><br>
                    <strong style="color:#10211d;">Recommended action:</strong> {safe_text(action)}
                </div>
            </div>
            """
        )

    st.caption("Satellite-to-pollution associations are screening evidence for investigation, not proof of a single source or causal relationship.")


# ============================================================
# PAGE: CITIZEN AI
# ============================================================

# ============================================================
# PAGE: CITIZEN AI
# ============================================================

def page_citizen():
    render_data_strip()

    render_hero(
        "CITIZEN REPORTING",
        "Turn observations into useful information.",
        (
            "Upload a photo, add a local sensor observation and a short description. "
            "Gemini analyses visible pollution indicators together with nearby monitoring, "
            "forecast and citizen sensor context."
        ),
        [
            "CITIZEN REPORTING",
            "LOCAL SENSOR",
            "GEMINI AI",
            "MULTIMODAL ANALYSIS",
        ],
    )

    station_lookup = (
        forecast_data[["station_id", "station"]]
        .drop_duplicates()
        .sort_values("station")
    )

    selected_station = st.selectbox(
        "Nearest monitoring station",
        station_lookup["station"].tolist(),
        key="citizen_nearest_station",
    )

    station_id = (
        station_lookup.loc[
            station_lookup["station"] == selected_station,
            "station_id",
        ]
        .iloc[0]
    )

    station_history = (
        forecast_data[
            forecast_data["station_id"] == station_id
        ]
        .sort_values("timestamp")
    )

    if station_history.empty:
        st.warning("No station context is available.")
        return

    station_row = station_history.iloc[-1]

    alert_rows = alerts[
        alerts["station"] == selected_station
    ]

    alert_row = (
        alert_rows.iloc[0]
        if not alert_rows.empty
        else None
    )

    hotspot_rows = hotspots[
        hotspots["station"] == selected_station
    ]

    hotspot_score = (
        float(
            hotspot_rows.iloc[0]["hotspot_score"]
        )
        if not hotspot_rows.empty
        else 0.0
    )

    render_section(
        "Environmental context",
        "The monitoring information supplied alongside the citizen observation.",
    )

    render_metrics(
        [
            {
                "label": "Current PM2.5",
                "value": f"{float(station_row['pm25']):.1f}",
                "support": "µg/m³ at nearest monitoring station",
            },
            {
                "label": "Predicted PM2.5",
                "value": (
                    f"{float(alert_row['predicted_pm25_3h']):.1f}"
                    if alert_row is not None
                    else "N/A"
                ),
                "support": "Three-hour model estimate",
            },
            {
                "label": "Hotspot score",
                "value": f"{hotspot_score:.2f}",
                "support": "Station-level detection score",
            },
            {
                "label": "Event links",
                "value": (
                    str(int(alert_row["recent_event_count"]))
                    if alert_row is not None
                    else "0"
                ),
                "support": "Recent satellite-linked observations",
            },
        ]
    )

    render_section(
        "Submit observation",
        (
            "Add a photo and local sensor reading from the citizen's observation point. "
            "Sensor fields accept manual or demo readings in this prototype and can later "
            "be connected to a physical device or sensor API."
        ),
    )

    with st.form(
        "citizen_report_form",
        clear_on_submit=False,
    ):

        uploaded_file = st.file_uploader(
            "Upload pollution image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear photo showing the environmental condition.",
        )

        sensor_source = st.selectbox(
            "Local sensor source",
            [
                "Citizen local sensor (user-provided device)",
                "Manual local measurement (entered by user)",
                "Demo sensor reading (prototype)",
            ],
            index=0,
            help=(
                "Identify the source of the local sensor values entered below. "
                "The prototype does not claim a live hardware connection."
            ),
        )

        sensor_col1, sensor_col2 = st.columns(2)

        with sensor_col1:

            citizen_pm25 = st.number_input(
                "Sensor PM2.5 (µg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=24.5,
                step=0.1,
                format="%.1f",
                help=(
                    "Local PM2.5 observation from the citizen "
                    "sensor or local measurement."
                ),
            )

            citizen_pm10 = st.number_input(
                "Sensor PM10 (µg/m³)",
                min_value=0.0,
                max_value=1500.0,
                value=46.0,
                step=0.1,
                format="%.1f",
                help="Optional local PM10 observation.",
            )

        with sensor_col2:

            citizen_temperature = st.number_input(
                "Temperature (°C)",
                min_value=-30.0,
                max_value=70.0,
                value=29.0,
                step=0.1,
                format="%.1f",
                help="Local temperature observation.",
            )

            citizen_humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=62.0,
                step=0.1,
                format="%.1f",
                help="Local relative humidity observation.",
            )

        sensor_timestamp = st.text_input(
            "Sensor observation timestamp",
            value=format_date(station_row["timestamp"]),
            help=(
                "Timestamp recorded by the local sensor or "
                "the observation entry."
            ),
        )

        report_text = st.text_area(
            "Describe what you observed",
            placeholder=(
                "Example: Visible smoke was observed near an "
                "industrial area during the afternoon."
            ),
            height=120,
        )

        submitted = st.form_submit_button(
            "Analyze with Gemini",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Observation preview",
            width="stretch",
        )

    if not submitted:
        return

    if uploaded_file is None:

        st.warning(
            "Please upload an image first."
        )

        return

    if not report_text.strip():

        st.warning(
            "Please provide a short description."
        )

        return

    # --------------------------------------------------------
    # DIRECT LOCAL SENSOR COMPARISON
    # --------------------------------------------------------

    sensor_delta = (
        float(citizen_pm25)
        - float(station_row["pm25"])
    )

    suffix = (
        Path(uploaded_file.name).suffix.lower()
        or ".jpg"
    )

    temp_file = NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )

    temp_path = Path(
        temp_file.name
    )

    try:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_file.close()

        with st.spinner(
            "Gemini is analysing the observation and local sensor context..."
        ):

            result = analyze_citizen_report(

                image_path=temp_path,

                report_text=report_text.strip(),

                latitude=float(
                    station_row["latitude"]
                ),

                longitude=float(
                    station_row["longitude"]
                ),

                current_pm25=float(
                    station_row["pm25"]
                ),

                forecast_pm25_3h=(

                    float(
                        alert_row[
                            "predicted_pm25_3h"
                        ]
                    )

                    if alert_row is not None

                    else None

                ),

                hotspot_score=hotspot_score,

                wind_speed=None,

                # ------------------------------------------------
                # CITIZEN SENSOR INPUT
                # ------------------------------------------------

                citizen_pm25=float(
                    citizen_pm25
                ),

                citizen_pm10=float(
                    citizen_pm10
                ),

                citizen_temperature_c=float(
                    citizen_temperature
                ),

                citizen_humidity_pct=float(
                    citizen_humidity
                ),

                citizen_sensor_timestamp=(
                    sensor_timestamp
                ),

                citizen_sensor_source=(
                    sensor_source
                ),

            )

        st.success(
            "AI visual assessment completed with citizen sensor context."
        )

    except Exception as exc:

        st.error(
            f"Gemini analysis failed: {exc}"
        )

        return

    finally:

        try:

            temp_path.unlink(
                missing_ok=True
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # CITIZEN SENSOR RESULTS
    # --------------------------------------------------------

    render_section(
        "Citizen sensor context",
        (
            "The submitted local sensor observation is shown "
            "alongside the nearest monitoring station value so "
            "the assessment has both local and network context."
        ),
    )

    render_metrics(
        [
            {
                "label": "Citizen PM2.5",
                "value": f"{citizen_pm25:.1f}",
                "support": "µg/m³ local observation",
            },

            {
                "label": "Nearest station",
                "value": (
                    f"{float(station_row['pm25']):.1f}"
                ),
                "support": "µg/m³ nearest monitoring value",
            },

            {
                "label": "Local difference",
                "value": (
                    f"+{sensor_delta:.1f}"
                    if sensor_delta >= 0
                    else f"{sensor_delta:.1f}"
                ),
                "support": (
                    "Citizen PM2.5 minus nearest station PM2.5"
                ),
            },

            {
                "label": "Sensor source",
                "value": sensor_source,
                "support": (
                    f"PM10 {citizen_pm10:.1f} · "
                    f"{citizen_temperature:.1f}°C · "
                    f"{citizen_humidity:.1f}%"
                ),
            },
        ]
    )

    st.html(
        f"""
        <div
            class="panel"
            style="margin-top:14px;"
        >

            <div class="panel-title">
                Local sensor contribution
            </div>

            <div
                style="
                    margin-top:10px;
                    color:#596863;
                    font-size:11px;
                    line-height:1.75;
                "
            >

                The citizen sensor reading is supplied as
                contextual evidence to the multimodal assessment.

                A difference from the nearest station can indicate
                local variation, but it does not by itself prove
                a pollution source.

                <br><br>

                Sensor values are citizen-provided observations
                and should be interpreted with the station,
                weather, forecast and satellite context.

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # GEMINI SENSOR INTERPRETATION
    # --------------------------------------------------------

    sensor_consistency = str(
        result.get(
            "sensor_consistency",
            "NO_SENSOR_DATA",
        )
    ).replace(
        "_",
        " ",
    )

    sensor_observations = result.get(
        "sensor_observations",
        [],
    )

    if sensor_observations:

        observation_html = (
            '<div style="margin-top:8px;">'
        )

        for observation in sensor_observations:

            observation_html += f"""
                <div
                    style="
                        padding:8px 0;
                        border-bottom:
                            1px solid
                            rgba(16,33,29,.06);
                        color:#596863;
                        font-size:11px;
                        line-height:1.65;
                    "
                >

                    <span
                        style="
                            color:#0f766e;
                            font-weight:900;
                            margin-right:7px;
                        "
                    >
                        +
                    </span>

                    {safe_text(observation)}

                </div>
            """

        observation_html += "</div>"

    else:

        observation_html = (
            '<div style="margin-top:8px;">'
            "No additional sensor observation was returned."
            "</div>"
        )

    st.html(
        f"""
        <div
            class="ai-box"
            style="margin-top:14px;"
        >

            <div class="ai-label">
                GEMINI SENSOR INTERPRETATION
            </div>

            <div class="ai-title">
                {safe_text(sensor_consistency)}
            </div>

            <div
                style="
                    margin-top:10px;
                    color:#596863;
                    font-size:11px;
                    line-height:1.75;
                "
            >

                {observation_html}

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # AI VISUAL ASSESSMENT
    # --------------------------------------------------------

    render_section(
        "AI visual assessment",
        (
            "Gemini interpretation of the visible evidence "
            "and supplied environmental context."
        ),
    )

    pollution_visible = (
        "YES"
        if result["pollution_visible"]
        else "NO"
    )

    category = (
        result["pollution_category"]
        .replace("_", " ")
        .title()
    )

    confidence = (
        float(result["confidence"])
        * 100
    )

    severity = str(
        result.get(
            "severity",
            "UNKNOWN",
        )
    )

    col1, col2 = st.columns(
        [1, 1.7],
        gap="large",
    )

    with col1:

        st.html(
            f"""
            <div class="ai-box">

                <div class="ai-label">
                    GEMINI AI
                </div>

                <div class="ai-title">
                    {safe_text(category)}
                </div>

                <div
                    class="alert-grid"
                    style="margin-top:18px;"
                >

                    <div class="alert-box">

                        <div class="alert-label">
                            Visible
                        </div>

                        <div class="alert-value">
                            {pollution_visible}
                        </div>

                    </div>

                    <div class="alert-box">

                        <div class="alert-label">
                            Confidence
                        </div>

                        <div class="alert-value">
                            {confidence:.0f}%
                        </div>

                    </div>

                    <div class="alert-box">

                        <div class="alert-label">
                            Severity
                        </div>

                        <div class="alert-value">
                            {safe_text(severity)}
                        </div>

                    </div>

                </div>

            </div>
            """
        )

    with col2:

        st.html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Assessment summary
                </div>

                <div
                    style="
                        margin-top:10px;
                        color:#596863;
                        font-size:12px;
                        line-height:1.75;
                    "
                >

                    {safe_text(
                        result.get(
                            "summary",
                            "No summary returned.",
                        )
                    )}

                </div>

                <div
                    style="
                        margin-top:16px;
                        padding:13px;
                        border-radius:13px;
                        background:#f1f7f5;
                    "
                >

                    <div
                        style="
                            font-size:10px;
                            text-transform:uppercase;
                            letter-spacing:.08em;
                            font-weight:800;
                            color:#0a5c56;
                        "
                    >
                        Recommended action
                    </div>

                    <div
                        style="
                            margin-top:6px;
                            color:#4d5f5a;
                            font-size:11px;
                            line-height:1.6;
                        "
                    >

                        {safe_text(
                            result.get(
                                "recommended_action",
                                "No recommendation returned.",
                            )
                        )}

                    </div>

                </div>

            </div>
            """
        )

    # --------------------------------------------------------
    # VISUAL EVIDENCE
    # --------------------------------------------------------

    render_section(
        "Visual evidence",
        (
            "Gemini describes visible indicators; the image "
            "assessment alone does not prove the exact pollution source."
        ),
    )

    evidence_html = (
        '<div class="panel">'
    )

    for evidence in result.get(
        "visual_evidence",
        [],
    ):

        evidence_html += f"""
            <div
                style="
                    padding:10px 0;
                    border-bottom:
                        1px solid rgba(16,33,29,.06);
                    font-size:12px;
                    color:#53635f;
                    line-height:1.6;
                "
            >

                <span
                    style="
                        color:#0f766e;
                        font-weight:900;
                        margin-right:7px;
                    "
                >
                    +
                </span>

                {safe_text(evidence)}

            </div>
        """

    evidence_html += "</div>"

    st.html(
        evidence_html
    )

    st.caption(
        (
            "Citizen sensor readings are contextual observations. "
            "They should be interpreted with nearby station, weather, "
            "forecast and satellite evidence rather than treated as "
            "proof of a pollution source."
        )
    )

# ============================================================
# PAGE: CLIMATE ACTION / FEDERATED NETWORK
# ============================================================

def page_model_network():
    render_data_strip()
    render_hero(
        "CLIMATE ACTION NETWORK",
        "Share intelligence, not raw data.",
        "Five regional India clients perform preprocessing and PM2.5 model training locally. Each client fits its own feature scaler using only its local training partition. The resulting model parameters are converted into a common raw-feature space and combined using sample-weighted FedAvg. The same client interface is designed for future BRICS-country expansion.",
        ["FEDERATED LEARNING", "LOCAL PREPROCESSING", "BRICS-READY"],
    )

    # --------------------------------------------------------
    # REAL FEDERATED IMPLEMENTATION STATUS
    # --------------------------------------------------------

    global_row = federated_global_metrics.iloc[0]

    clients = int(global_row.get("clients", len(federated_client_metrics)))
    training_samples = int(global_row.get("training_samples", 0))
    test_samples = int(global_row.get("test_samples", 0))
    feature_count = int(global_row.get("feature_count", len(federated_bundle.get("features", []))))
    global_mae = float(global_row.get("mae_ug_m3", 0) or 0)
    global_rmse = float(global_row.get("rmse_ug_m3", 0) or 0)
    global_r2 = float(global_row.get("r2", 0) or 0)

    render_section(
        "Federated implementation",
        "These values come from the completed regional training run and saved model artifacts.",
    )
    render_metrics(
        [
            {
                "label": "Regional clients",
                "value": str(clients),
                "support": "India regional training clients",
            },
            {
                "label": "Local training rows",
                "value": f"{training_samples:,}",
                "support": "Rows retained across local clients",
            },
            {
                "label": "FedAvg",
                "value": "ACTIVE",
                "support": "Sample-weighted parameter aggregation",
            },
            {
                "label": "Raw X / y transfer",
                "value": "NO",
                "support": "Only model parameters and sample counts are aggregated",
            },
        ]
    )

    render_section(
        "Shared model evaluation",
        "The aggregated model is evaluated on a chronological holdout that was not used for local client training.",
    )
    render_metrics(
        [
            {
                "label": "MAE",
                "value": f"{global_mae:.2f}",
                "support": "µg/m³ on federated test data",
            },
            {
                "label": "RMSE",
                "value": f"{global_rmse:.2f}",
                "support": "µg/m³ on federated test data",
            },
            {
                "label": "R²",
                "value": f"{global_r2:.3f}",
                "support": f"{test_samples:,} chronological test rows",
            },
            {
                "label": "Features",
                "value": str(feature_count),
                "support": "Shared model feature interface",
            },
        ]
    )

    # --------------------------------------------------------
    # REGIONAL CLIENTS
    # --------------------------------------------------------

    render_section(
        "Regional client network",
        "The current implementation uses five geographic India clients. These are technical training partitions, not official administrative boundaries.",
    )

    client_display = federated_client_metrics.copy()
    client_display = client_display.rename(
        columns={
            "client_id": "Regional client",
            "samples": "Local samples",
            "stations": "Stations",
            "train_mae_ug_m3": "Local MAE",
            "train_rmse_ug_m3": "Local RMSE",
        }
    )

    preferred_columns = [
        "Regional client",
        "Local samples",
        "Stations",
        "Local MAE",
        "Local RMSE",
    ]
    client_display = client_display[[
        column for column in preferred_columns
        if column in client_display.columns
    ]].copy()

    if not client_display.empty:
        for column in ["Local MAE", "Local RMSE"]:
            if column in client_display.columns:
                client_display[column] = client_display[column].map(
                    lambda value: f"{float(value):.2f}"
                )

        st.dataframe(
            client_display,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # FEDERATED DATA FLOW
    # --------------------------------------------------------

    render_section(
        "How the coordination layer works",
        "The application demonstrates local preprocessing, local model training and parameter aggregation while keeping the underlying environmental rows at each client.",
    )
    render_features(
        [
            {
                "icon": "01",
                "title": "Local data",
                "copy": "Each regional client retains its own air-quality training observations locally.",
            },
            {
                "icon": "02",
                "title": "Local training",
                "copy": "Each client fits its own feature scaler and trains the same compatible PM2.5 forecasting model using only its local training partition.",
            },
            {
                "icon": "03",
                "title": "Parameter exchange",
                "copy": "The locally trained model is converted into a common raw-feature parameter space before model parameters and sample count are prepared for aggregation.",
            },
            {
                "icon": "04",
                "title": "FedAvg",
                "copy": "The coordinator performs sample-weighted Federated Averaging in the common raw-feature space.",
            },
            {
                "icon": "05",
                "title": "Shared model",
                "copy": "The aggregated model can be evaluated region by region and redistributed as a common model artifact.",
            },
            {
                "icon": "06",
                "title": "BRICS expansion",
                "copy": "The same client interface can accept another jurisdiction's local provider implementations without moving its raw environmental data into the shared layer.",
            },
        ]
    )


    # --------------------------------------------------------
    # ECONOMIC CORRIDOR INTELLIGENCE
    # --------------------------------------------------------

    corridor_file = (
        PROJECT_ROOT / "data/processed/corridor_intelligence.csv"
    )

    corridor_station_file = (
        PROJECT_ROOT / "data/processed/corridor_station_signals.csv"
    )

    if corridor_file.exists():

        corridor_intelligence = pd.read_csv(
            corridor_file
        )

        if corridor_station_file.exists():

            corridor_station_signals = pd.read_csv(
                corridor_station_file
            )

        else:

            corridor_station_signals = pd.DataFrame()

        render_section(
            "Economic corridor intelligence",
            (
                "Potential corridor-level signals combine current "
                "PM2.5, three-hour forecasts, hotspot indicators, "
                "recent satellite-linked event signals and authority "
                "alerts. Monitoring coverage is shown separately."
            ),
        )

        # ----------------------------------------------------
        # COVERAGE SUMMARY
        # ----------------------------------------------------

        corridor_count = int(
            corridor_intelligence["corridor_id"].nunique()
        )

        coverage_series = pd.to_numeric(
            corridor_intelligence["forecast_coverage_pct"],
            errors="coerce",
        )

        high_coverage_count = int(
            (coverage_series >= 80).sum()
        )

        medium_coverage_count = int(
            (
                (coverage_series >= 50)
                & (coverage_series < 80)
            ).sum()
        )

        limited_coverage_count = int(
            (coverage_series < 50).sum()
        )

        render_metrics(
            [
                {
                    "label": "Corridors monitored",
                    "value": str(corridor_count),
                    "support": "Configured economic corridor views",
                },
                {
                    "label": "High coverage",
                    "value": str(high_coverage_count),
                    "support": "At least 80% forecast coverage",
                },
                {
                    "label": "Medium coverage",
                    "value": str(medium_coverage_count),
                    "support": "50%–79.9% forecast coverage",
                },
                {
                    "label": "Limited coverage",
                    "value": str(limited_coverage_count),
                    "support": "Below 50% forecast coverage",
                },
            ]
        )

        corridor_name_map = {
            "DMIC": "Delhi–Mumbai Industrial Corridor",
            "AKIC": "Amritsar–Kolkata Industrial Corridor",
            "CBIC": "Chennai–Bengaluru Industrial Corridor",
            "VCIC": "Vizag–Chennai Industrial Corridor",
            "ECEC": "East Coast Industrial Corridor",
            "BMIC": "Bengaluru–Mumbai Industrial Corridor",
        }

        corridor_display_id_map = {
            "ECEC": "ECIC",
        }

        corridor_order = [
            "DMIC",
            "AKIC",
            "CBIC",
            "VCIC",
            "ECEC",
            "BMIC",
        ]

        # ----------------------------------------------------
        # CORRIDOR OVERVIEW CARDS
        # ----------------------------------------------------

        corridor_cards = []

        corridor_sorted = corridor_intelligence.copy()
        corridor_sorted["_order"] = (
            corridor_sorted["corridor_id"]
            .map(
                {
                    value: index
                    for index, value
                    in enumerate(corridor_order)
                }
            )
            .fillna(99)
        )
        corridor_sorted = corridor_sorted.sort_values("_order")

        for _, row in corridor_sorted.iterrows():

            raw_id = str(
                row.get(
                    "corridor_id",
                    "—",
                )
            )

            display_id = corridor_display_id_map.get(
                raw_id,
                raw_id,
            )

            corridor_name = corridor_name_map.get(
                raw_id,
                str(
                    row.get(
                        "corridor_name",
                        "Economic corridor",
                    )
                ),
            )

            signal_level = str(
                row.get(
                    "signal_level",
                    "UNKNOWN",
                )
            )

            coverage_label = str(
                row.get(
                    "coverage_label",
                    "UNKNOWN",
                )
            ).replace(
                "_",
                " ",
            )

            score = pd.to_numeric(
                row.get(
                    "signal_score",
                    None,
                ),
                errors="coerce",
            )

            coverage = pd.to_numeric(
                row.get(
                    "forecast_coverage_pct",
                    None,
                ),
                errors="coerce",
            )

            current_pm25 = pd.to_numeric(
                row.get(
                    "mean_current_pm25",
                    None,
                ),
                errors="coerce",
            )

            forecast_pm25 = pd.to_numeric(
                row.get(
                    "mean_forecast_pm25_3h",
                    None,
                ),
                errors="coerce",
            )

            forecast_change = pd.to_numeric(
                row.get(
                    "mean_forecast_change",
                    None,
                ),
                errors="coerce",
            )

            associated = pd.to_numeric(
                row.get(
                    "associated_stations",
                    0,
                ),
                errors="coerce",
            )

            hotspots_count = pd.to_numeric(
                row.get(
                    "hotspot_station_count",
                    0,
                ),
                errors="coerce",
            )

            event_count = pd.to_numeric(
                row.get(
                    "event_station_count",
                    0,
                ),
                errors="coerce",
            )

            authority_count = pd.to_numeric(
                row.get(
                    "authority_alert_station_count",
                    0,
                ),
                errors="coerce",
            )

            if signal_level == "HIGH":
                badge_color = "#8b3f36"
                badge_background = "#f7e9e6"
            elif signal_level == "MODERATE":
                badge_color = "#946d1c"
                badge_background = "#faf3df"
            elif signal_level == "LIMITED_COVERAGE":
                badge_color = "#6f6b61"
                badge_background = "#eceae4"
            else:
                badge_color = "#2f6f57"
                badge_background = "#e8f2ed"

            corridor_cards.append(
                f"""
                <div class="node">

                    <div class="node-line"></div>

                    <div
                        style="
                            display:flex;
                            align-items:flex-start;
                            justify-content:space-between;
                            gap:10px;
                        "
                    >

                        <div>

                            <div class="node-name">
                                {safe_text(display_id)}
                            </div>

                            <div
                                class="node-state"
                                style="
                                    margin-top:4px;
                                    line-height:1.45;
                                    color:#6f7d79;
                                "
                            >
                                {safe_text(corridor_name)}
                            </div>

                            {
                                '<div style="margin-top:3px;color:#6f7d79;font-size:9px;">Phase 1 of ECIC</div>'
                                if raw_id == "VCIC"
                                else ""
                            }

                        </div>

                        <div
                            style="
                                flex-shrink:0;
                                padding:5px 8px;
                                border-radius:999px;
                                background:{badge_background};
                                color:{badge_color};
                                font-size:8px;
                                font-weight:900;
                                letter-spacing:.05em;
                                text-transform:uppercase;
                            "
                        >
                            {safe_text(signal_level)}
                        </div>

                    </div>

                    <div
                        style="
                            margin-top:14px;
                            display:grid;
                            grid-template-columns:repeat(2,minmax(0,1fr));
                            gap:10px;
                        "
                    >

                        <div>
                            <div style="color:#7a8884;font-size:8px;letter-spacing:.07em;text-transform:uppercase;">
                                Signal index
                            </div>
                            <div style="margin-top:4px;color:#17322d;font-size:20px;font-weight:800;">
                                {f"{score:.1f}" if pd.notna(score) else "—"}
                            </div>
                        </div>

                        <div>
                            <div style="color:#7a8884;font-size:8px;letter-spacing:.07em;text-transform:uppercase;">
                                Forecast coverage
                            </div>
                            <div style="margin-top:4px;color:#17322d;font-size:20px;font-weight:800;">
                                {f"{coverage:.1f}%" if pd.notna(coverage) else "—"}
                            </div>
                        </div>

                    </div>

                    <div style="margin-top:12px;color:#596863;font-size:10px;line-height:1.7;">

                        <strong>Current:</strong>
                        {f"{current_pm25:.1f}" if pd.notna(current_pm25) else "—"}
                        µg/m³

                        ·

                        <strong>3h:</strong>
                        {f"{forecast_pm25:.1f}" if pd.notna(forecast_pm25) else "—"}
                        µg/m³

                        <br>

                        <strong>Change:</strong>
                        {f"{forecast_change:+.1f}" if pd.notna(forecast_change) else "—"}
                        µg/m³

                    </div>

                    <div style="margin-top:9px;color:#6f7d79;font-size:9px;line-height:1.6;">

                        {int(associated) if pd.notna(associated) else 0} stations
                        ·
                        {int(hotspots_count) if pd.notna(hotspots_count) else 0} hotspots
                        ·
                        {int(event_count) if pd.notna(event_count) else 0} events
                        ·
                        {int(authority_count) if pd.notna(authority_count) else 0} authority alerts

                    </div>

                    <div style="margin-top:9px;color:#7a8884;font-size:8px;letter-spacing:.07em;text-transform:uppercase;">
                        {safe_text(coverage_label)}
                    </div>

                </div>
                """
            )

        st.html(
            '<div class="node-grid" style="margin-top:14px;">'
            + "".join(corridor_cards)
            + "</div>"
        )

        # ----------------------------------------------------
        # CORRIDOR SUMMARY TABLE
        # ----------------------------------------------------

        corridor_table = corridor_intelligence.copy()

        corridor_table["corridor_id"] = (
            corridor_table["corridor_id"]
            .replace(
                {
                    "ECEC": "ECIC",
                }
            )
        )

        corridor_table = corridor_table.rename(
            columns={
                "corridor_id": "Corridor",
                "forecast_coverage_pct": "Forecast coverage %",
                "mean_current_pm25": "Current PM2.5",
                "mean_forecast_pm25_3h": "3h forecast PM2.5",
                "mean_forecast_change": "Forecast change",
                "hotspot_station_count": "Hotspot stations",
                "event_station_count": "Event stations",
                "authority_alert_station_count": "Authority alerts",
                "signal_score": "Signal index",
                "signal_level": "Signal level",
            }
        )

        corridor_table = corridor_table[
            [
                column
                for column in [
                    "Corridor",
                    "Forecast coverage %",
                    "Current PM2.5",
                    "3h forecast PM2.5",
                    "Forecast change",
                    "Hotspot stations",
                    "Event stations",
                    "Authority alerts",
                    "Signal index",
                    "Signal level",
                ]
                if column in corridor_table.columns
            ]
        ]

        st.dataframe(
            corridor_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Forecast coverage %":
                    st.column_config.NumberColumn(format="%.1f%%"),
                "Current PM2.5":
                    st.column_config.NumberColumn(format="%.1f"),
                "3h forecast PM2.5":
                    st.column_config.NumberColumn(format="%.1f"),
                "Forecast change":
                    st.column_config.NumberColumn(format="%+.1f"),
                "Signal index":
                    st.column_config.NumberColumn(format="%.1f"),
            },
        )

        # ----------------------------------------------------
        # INTERACTIVE CORRIDOR DRILL-DOWN
        # ----------------------------------------------------

        if not corridor_station_signals.empty:

            render_section(
                "Corridor signal investigation",
                "Select a corridor to inspect the station-level indicators contributing to its monitoring signal.",
            )

            available_ids = set(
                corridor_station_signals[
                    "corridor_id"
                ]
                .dropna()
                .astype(str)
            )

            corridor_options = [
                corridor_id
                for corridor_id in corridor_order
                if corridor_id in available_ids
            ]

            if corridor_options:

                def corridor_format(corridor_id):

                    display_id = corridor_display_id_map.get(
                        corridor_id,
                        corridor_id,
                    )

                    label = (
                        f"{display_id} · "
                        f"{corridor_name_map.get(corridor_id, corridor_id)}"
                    )

                    if corridor_id == "VCIC":
                        label += " · Phase 1 of ECIC"

                    return label

                selected_corridor = st.selectbox(
                    "Select corridor",
                    corridor_options,
                    format_func=corridor_format,
                    key="corridor_signal_selector",
                    help=(
                        "Monitoring view based on participating stations; "
                        "not a route-wide diagnosis."
                    ),
                )

                selected_station_rows = corridor_station_signals[
                    corridor_station_signals[
                        "corridor_id"
                    ].astype(str)
                    == str(selected_corridor)
                ].copy()

                selected_summary_rows = corridor_intelligence[
                    corridor_intelligence[
                        "corridor_id"
                    ].astype(str)
                    == str(selected_corridor)
                ].copy()

                if (
                    not selected_station_rows.empty
                    and not selected_summary_rows.empty
                ):

                    summary_row = selected_summary_rows.iloc[0]

                    selected_score = pd.to_numeric(
                        summary_row.get(
                            "signal_score"
                        ),
                        errors="coerce",
                    )

                    selected_coverage = pd.to_numeric(
                        summary_row.get(
                            "forecast_coverage_pct"
                        ),
                        errors="coerce",
                    )

                    selected_current = pd.to_numeric(
                        summary_row.get(
                            "mean_current_pm25"
                        ),
                        errors="coerce",
                    )

                    selected_forecast = pd.to_numeric(
                        summary_row.get(
                            "mean_forecast_pm25_3h"
                        ),
                        errors="coerce",
                    )

                    selected_change = pd.to_numeric(
                        summary_row.get(
                            "mean_forecast_change"
                        ),
                        errors="coerce",
                    )

                    selected_signal_level = str(
                        summary_row.get(
                            "signal_level",
                            "UNKNOWN",
                        )
                    )

                    selected_display_id = corridor_display_id_map.get(
                        selected_corridor,
                        selected_corridor,
                    )

                    selected_label = (
                        f"{selected_display_id} · "
                        f"{corridor_name_map.get(selected_corridor, selected_corridor)}"
                    )

                    if selected_corridor == "VCIC":
                        selected_label += " · Phase 1 of ECIC"

                    render_metrics(
                        [
                            {
                                "label": "Signal index",
                                "value": (
                                    f"{selected_score:.1f}"
                                    if pd.notna(selected_score)
                                    else "—"
                                ),
                                "support": "Project-defined monitoring composite",
                            },
                            {
                                "label": "Forecast coverage",
                                "value": (
                                    f"{selected_coverage:.1f}%"
                                    if pd.notna(selected_coverage)
                                    else "—"
                                ),
                                "support": "Forecast coverage across participating stations",
                            },
                            {
                                "label": "Mean current PM2.5",
                                "value": (
                                    f"{selected_current:.1f}"
                                    if pd.notna(selected_current)
                                    else "—"
                                ),
                                "support": "µg/m³ across participating stations",
                            },
                            {
                                "label": "Mean forecast change",
                                "value": (
                                    f"{selected_change:+.1f}"
                                    if pd.notna(selected_change)
                                    else "—"
                                ),
                                "support": "Current to approximately three hours ahead",
                            },
                        ]
                    )

                    # --------------------------------------------
                    # EVIDENCE LABELS
                    # --------------------------------------------

                    def evidence_label(row):

                        evidence = []

                        if as_bool(
                            row.get(
                                "current_pm25_elevated",
                                False,
                            )
                        ):
                            evidence.append(
                                "Current PM2.5"
                            )

                        if as_bool(
                            row.get(
                                "forecast_pm25_elevated",
                                False,
                            )
                        ):
                            evidence.append(
                                "3h forecast"
                            )

                        if as_bool(
                            row.get(
                                "hotspot_signal",
                                False,
                            )
                        ):
                            evidence.append(
                                "Hotspot"
                            )

                        if as_bool(
                            row.get(
                                "event_signal",
                                False,
                            )
                        ):
                            evidence.append(
                                "Satellite event"
                            )

                        if as_bool(
                            row.get(
                                "authority_pressure",
                                False,
                            )
                        ):
                            evidence.append(
                                "Authority alert"
                            )

                        if not evidence:
                            return "Monitoring only"

                        return " · ".join(evidence)

                    selected_station_rows["Evidence"] = (
                        selected_station_rows.apply(
                            evidence_label,
                            axis=1,
                        )
                    )

                    station_display = selected_station_rows.rename(
                        columns={
                            "station": "Station",
                            "state": "State",
                            "pm25": "Current PM2.5",
                            "predicted_pm25_3h": "3h forecast",
                            "forecast_change": "Forecast change",
                            "hotspot_score": "Hotspot score",
                            "recent_event_count": "Event links",
                            "forecast_risk": "Forecast risk",
                            "alert_priority": "Authority priority",
                        }
                    )

                    station_columns = [
                        "Station",
                        "State",
                        "Current PM2.5",
                        "3h forecast",
                        "Forecast change",
                        "Hotspot score",
                        "Event links",
                        "Forecast risk",
                        "Authority priority",
                        "Evidence",
                    ]

                    station_display = station_display[
                        [
                            column
                            for column in station_columns
                            if column in station_display.columns
                        ]
                    ].copy()

                    for column in [
                        "Current PM2.5",
                        "3h forecast",
                        "Forecast change",
                        "Hotspot score",
                    ]:

                        if column in station_display.columns:

                            station_display[column] = pd.to_numeric(
                                station_display[column],
                                errors="coerce",
                            )

                    if "Event links" in station_display.columns:

                        station_display["Event links"] = (
                            pd.to_numeric(
                                station_display["Event links"],
                                errors="coerce",
                            )
                            .fillna(0)
                            .astype(int)
                        )

                    st.dataframe(
                        station_display,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Current PM2.5":
                                st.column_config.NumberColumn(format="%.1f"),
                            "3h forecast":
                                st.column_config.NumberColumn(format="%.1f"),
                            "Forecast change":
                                st.column_config.NumberColumn(format="%+.1f"),
                            "Hotspot score":
                                st.column_config.NumberColumn(format="%.2f"),
                        },
                    )

                    st.html(
                        f"""
                        <div class="panel" style="margin-top:14px;">

                            <div class="panel-title">
                                How to read {safe_text(selected_label)}
                            </div>

                            <div
                                style="
                                    margin-top:10px;
                                    color:#596863;
                                    font-size:11px;
                                    line-height:1.75;
                                "
                            >

                                A station can contribute more than one
                                evidence signal. The Evidence column
                                shows the indicators available for
                                investigation at that station.

                                <br><br>

                                The corridor signal is a project-defined
                                monitoring composite. It does not establish
                                that a particular source caused observed
                                pollution and does not imply continuous
                                observation across the entire corridor.

                            </div>

                        </div>
                        """
                    )

                else:

                    st.info(
                        "No station-level signals are available for the selected corridor."
                    )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.html(
            """
            <div
                class="ai-box"
                style="margin-top:14px;"
            >

                <div class="ai-label">
                    CORRIDOR INTELLIGENCE
                </div>

                <div class="ai-title">
                    Monitoring signal, not route-wide diagnosis
                </div>

                <div style="
                    margin-top:12px;
                    color:#596863;
                    font-size:11px;
                    line-height:1.75;
                ">

                    The signal index is a project-defined composite
                    of available station-level indicators; it is
                    not a validated regulatory risk score.

                    <br><br>

                    Corridor summaries reflect participating
                    monitoring locations and do not imply continuous
                    observation across every point of a corridor.

                    <br><br>

                    Satellite-linked events are screening evidence
                    for investigation and do not by themselves
                    establish a single pollution source or causal
                    relationship.

                </div>

            </div>
            """
        )

    else:

        st.info(
            "Corridor intelligence data is not available. Run the corridor intelligence pipeline first."
        )


    # --------------------------------------------------------
    # BRICS NODE READINESS
    # --------------------------------------------------------

    render_section(
        "BRICS interoperability",
        "India is the demonstrated implementation. Other BRICS countries are shown as compatible expansion nodes, not as live participating datasets.",
    )

    nodes = [
        ("India", "Active regional implementation"),
        ("Brazil", "Compatible expansion node"),
        ("South Africa", "Compatible expansion node"),
        ("China", "Compatible expansion node"),
        ("Russia", "Compatible expansion node"),
    ]

    html = '<div class="node-grid">'
    for name, state in nodes:
        html += f"""
        <div class="node">
            <div class="node-line"></div>
            <div class="node-name">{safe_text(name)}</div>
            <div class="node-state">● {safe_text(state)}</div>
        </div>
        """
    html += "</div>"
    st.html(html)

    # --------------------------------------------------------
    # IMPLEMENTATION LIMITATION
    # --------------------------------------------------------

    st.info(
        "Current deployment: five regional India clients using the federated training pipeline. This is a working federated-learning demonstration, not a live intergovernmental BRICS network. Production deployment would additionally require authenticated participants, secure aggregation, and operational governance between institutions."
    )

    render_section(
        "Evidence and terminology",
        "Event intelligence is intentionally framed as screening evidence rather than definitive pollution-source attribution.",
    )
    render_features(
        [
            {
                "icon": "E1",
                "title": "Potential event",
                "copy": "Use evidence-based language such as potential agricultural-burning, industrial-related or regional-transport event.",
            },
            {
                "icon": "E2",
                "title": "Evidence chain",
                "copy": "Combine satellite detections, land-cover context, PM2.5 anomalies, weather consistency and forecast signals before escalating review.",
            },
            {
                "icon": "E3",
                "title": "Human review",
                "copy": "Authority alerts are decision-support signals for investigation and response, not automatic proof of a pollution source.",
            },
        ]
    )


# ============================================================
# TOP HEADER + RIGHT-ALIGNED NAVIGATION
# ============================================================

def render_top_header(overview_page, air_quality_page, events_page, citizen_page, model_network_page):
    st.html('<div class="app-top-safe"></div>')
    brand_col, overview_col, air_col, events_col, citizen_col, action_col = st.columns(
        [5.45, 1.0, 1.15, 1.25, 1.0, 1.18],
        gap="small",
    )

    with brand_col:
        st.html(
            f"""
            <div class="brand" style="margin:0;">
                <div class="brand-left">
                    <div class="brand-mark">
                        <svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                            <path d="M22.6 7.1C18.2 7.5 13.8 9.1 11 12.2C8.9 14.5 8.3 17.4 9.9 19.7C11.7 22.3 15.1 22.7 17.8 20.9C20.9 18.9 22.4 14.2 22.6 7.1Z" fill="white" fill-opacity="0.96"/>
                            <path d="M9.1 24.8C12.2 20.4 15.8 17.1 21.8 13.1" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
                            <path d="M6 11.4C8.3 9.6 10.4 9 13 9.2" stroke="white" stroke-opacity="0.62" stroke-width="1.6" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <div class="brand-name">{safe_text(PROJECT_NAME)}</div>
                        <div class="brand-subtitle">{safe_text(PROJECT_SUBTITLE)}</div>
                    </div>
                </div>
            </div>
            """
        )

    with overview_col:
        st.page_link(overview_page, label="Overview", icon=":material/dashboard:", width="content")
    with air_col:
        st.page_link(air_quality_page, label="Air Quality", icon=":material/air:", width="content")
    with events_col:
        st.page_link(events_page, label="Hotspot Intelligence", icon=":material/radar:", width="content")
    with citizen_col:
        st.page_link(citizen_page, label="Citizen AI", icon=":material/auto_awesome:", width="content")
    with action_col:
        st.page_link(model_network_page, label="Climate Action", icon=":material/hub:", width="content")


# ============================================================
# NAVIGATION
# ============================================================
#
# Streamlit's native top navigation is intentionally hidden here.
# The five Page objects remain registered with st.navigation, while
# st.page_link renders the custom navigation inside the brand header.
# This gives us a true top-right layout without radio buttons.

overview_page = st.Page(
    page_overview,
    title="Overview",
    icon=":material/dashboard:",
    default=True,
    url_path="overview",
)
air_quality_page = st.Page(
    page_air_quality,
    title="Air Quality",
    icon=":material/air:",
    url_path="air-quality",
)
events_page = st.Page(
    page_events,
    title="Pollution Events",
    icon=":material/radar:",
    url_path="events",
)
citizen_page = st.Page(
    page_citizen,
    title="Citizen AI",
    icon=":material/auto_awesome:",
    url_path="citizen-ai",
)
model_network_page = st.Page(
    page_model_network,
    title="Climate Action",
    icon=":material/hub:",
    url_path="climate-action",
)

pages = [
    overview_page,
    air_quality_page,
    events_page,
    citizen_page,
    model_network_page,
]

# Keep Streamlit's routing active but hide its native menu.
# The visible menu is rendered manually in the application header.
pg = st.navigation(pages, position="hidden")
render_top_header(overview_page, air_quality_page, events_page, citizen_page, model_network_page)
pg.run()
