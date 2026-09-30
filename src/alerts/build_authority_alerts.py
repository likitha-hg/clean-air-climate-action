from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_DATASET_FILE = (
    PROJECT_ROOT
    / "data/processed/pm25_forecast_dataset.csv"
)

HOTSPOT_FILE = (
    PROJECT_ROOT
    / "data/processed/hotspots_latest.csv"
)

EVENT_FILE = (
    PROJECT_ROOT
    / "data/processed/pollution_event_intelligence_with_forecast.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models/pm25_xgb_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "authority_alerts_final.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HIGH_FORECAST_THRESHOLD_PERCENTILE = 0.90
MODERATE_FORECAST_THRESHOLD_PERCENTILE = 0.75

RECENT_EVENT_WINDOW_HOURS = 48

# Event evidence needed before a forecast can elevate
# a station to HIGH.
HIGH_EVENT_SUPPORT_SCORE = 0.60


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    bundle = joblib.load(
        MODEL_FILE
    )

    if not isinstance(bundle, dict):
        raise TypeError(
            "Saved model is not a model bundle dictionary."
        )

    if "model" not in bundle:
        raise KeyError(
            "Model bundle missing 'model'."
        )

    if "features" not in bundle:
        raise KeyError(
            "Model bundle missing 'features'."
        )

    return (
        bundle["model"],
        bundle["features"],
    )


# ============================================================
# LOAD FORECAST DATA
# ============================================================

def load_forecast_data():

    if not FORECAST_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found: "
            f"{FORECAST_DATASET_FILE}"
        )

    df = pd.read_csv(
        FORECAST_DATASET_FILE,
        parse_dates=["timestamp"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=[
            "station_id",
            "timestamp",
            "pm25",
            "latitude",
            "longitude",
        ]
    ).copy()

    return df


# ============================================================
# LOAD HOTSPOTS
# ============================================================

def load_hotspots():

    if not HOTSPOT_FILE.exists():
        raise FileNotFoundError(
            f"Hotspot file not found: {HOTSPOT_FILE}"
        )

    hotspots = pd.read_csv(
        HOTSPOT_FILE
    )

    required = [
        "station_id",
        "hotspot_score",
        "risk_level",
    ]

    missing = [
        column
        for column in required
        if column not in hotspots.columns
    ]

    if missing:
        raise ValueError(
            f"Hotspot file missing columns: {missing}"
        )

    return hotspots[
        required
    ].drop_duplicates(
        "station_id"
    )


# ============================================================
# LOAD HISTORICAL EVENT INTELLIGENCE
# ============================================================

def load_event_intelligence():

    if not EVENT_FILE.exists():
        return pd.DataFrame()

    events = pd.read_csv(
        EVENT_FILE
    )

    required = [
        "fire_event_id",
        "station_id",
        "station",
        "fire_timestamp",
        "source_context",
        "event_score",
        "integrated_event_score",
        "integrated_priority",
        "evidence_level",
    ]

    missing = [
        column
        for column in required
        if column not in events.columns
    ]

    if missing:
        print(
            "Warning: event intelligence is missing:",
            missing,
        )
        return pd.DataFrame()

    events["fire_timestamp"] = pd.to_datetime(
        events["fire_timestamp"],
        errors="coerce",
        utc=True,
    )

    numeric_columns = [
        "event_score",
        "integrated_event_score",
    ]

    for column in numeric_columns:
        events[column] = pd.to_numeric(
            events[column],
            errors="coerce",
        )

    events = events.dropna(
        subset=[
            "station_id",
            "fire_timestamp",
        ]
    ).copy()

    return events


# ============================================================
# CALCULATE FORECAST THRESHOLDS
# ============================================================

def calculate_forecast_thresholds(
    forecast_data,
):
    pm25 = forecast_data[
        "pm25"
    ].dropna()

    if pm25.empty:
        raise ValueError(
            "No PM2.5 observations available."
        )

    q75 = pm25.quantile(
        MODERATE_FORECAST_THRESHOLD_PERCENTILE
    )

    q90 = pm25.quantile(
        HIGH_FORECAST_THRESHOLD_PERCENTILE
    )

    return q75, q90


# ============================================================
# GENERATE LATEST FORECASTS
# ============================================================

def generate_latest_forecasts(
    forecast_data,
    model,
    model_features,
):
    """
    Produce one forecast for the latest available model-input
    record at each station.

    Important:
    This is the latest forecast possible from the locally
    available dataset. It is not necessarily real-time.
    """

    missing = [
        column
        for column in model_features
        if column not in forecast_data.columns
    ]

    if missing:
        raise ValueError(
            f"Forecast dataset missing model features: {missing}"
        )

    latest = (
        forecast_data
        .sort_values("timestamp")
        .groupby(
            "station_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    complete = latest.dropna(
        subset=model_features
    ).copy()

    if complete.empty:
        raise ValueError(
            "No latest station records contain "
            "all model features."
        )

    complete[
        "predicted_pm25_3h"
    ] = np.maximum(
        model.predict(
            complete[
                model_features
            ]
        ),
        0,
    )

    complete[
        "forecast_target_timestamp"
    ] = (
        complete[
            "timestamp"
        ]
        + pd.Timedelta(
            hours=3
        )
    )

    return complete


# ============================================================
# FORECAST RISK
# ============================================================

def forecast_risk(
    value,
    q75,
    q90,
):
    if pd.isna(value):
        return "UNKNOWN"

    value = float(value)

    if value >= q90:
        return "HIGH"

    if value >= q75:
        return "MODERATE"

    return "LOW"


# ============================================================
# RECENT EVENT EVIDENCE
# ============================================================

def build_recent_event_summary(
    station_id,
    snapshot_time,
    event_data,
):
    """
    Find the strongest event evidence for this station in the
    recent event window.
    """

    if event_data.empty:

        return {
            "recent_event_count": 0,
            "recent_event_score": 0.0,
            "recent_source_context": "NONE",
            "recent_event_priority": "NONE",
            "recent_event_evidence": "NONE",
        }

    start_time = (
        snapshot_time
        - pd.Timedelta(
            hours=RECENT_EVENT_WINDOW_HOURS
        )
    )

    station_events = event_data[
        event_data["station_id"] == station_id
    ].copy()

    station_events = station_events[
        (station_events["fire_timestamp"] >= start_time)
        & (
            station_events["fire_timestamp"]
            <= snapshot_time
        )
    ].copy()

    if station_events.empty:

        return {
            "recent_event_count": 0,
            "recent_event_score": 0.0,
            "recent_source_context": "NONE",
            "recent_event_priority": "NONE",
            "recent_event_evidence": "NONE",
        }

    station_events = station_events.sort_values(
        "integrated_event_score",
        ascending=False,
    )

    strongest = station_events.iloc[0]

    return {
        "recent_event_count": len(
            station_events
        ),

        "recent_event_score": float(
            strongest[
                "integrated_event_score"
            ]
        ),

        "recent_source_context": str(
            strongest[
                "source_context"
            ]
        ),

        "recent_event_priority": str(
            strongest[
                "integrated_priority"
            ]
        ),

        "recent_event_evidence": str(
            strongest[
                "evidence_level"
            ]
        ),
    }


# ============================================================
# ALERT PRIORITY
# ============================================================

def calculate_alert_priority(
    hotspot_risk,
    forecast_risk_value,
    hotspot_score,
    forecast_value,
    current_pm25,
    recent_event_score,
    recent_event_count,
):
    """
    Conservative operational priority logic.

    A forecast alone should not automatically create a
    critical alert. Strong current/event evidence is required.
    """

    hotspot_risk = str(
        hotspot_risk
    )

    forecast_risk_value = str(
        forecast_risk_value
    )

    hotspot_score = float(
        hotspot_score
        if not pd.isna(
            hotspot_score
        )
        else 0.0
    )

    forecast_value = float(
        forecast_value
        if not pd.isna(
            forecast_value
        )
        else 0.0
    )

    current_pm25 = float(
        current_pm25
        if not pd.isna(
            current_pm25
        )
        else 0.0
    )

    recent_event_score = float(
        recent_event_score
        if not pd.isna(
            recent_event_score
        )
        else 0.0
    )

    # --------------------------------------------------------
    # CRITICAL REVIEW
    # --------------------------------------------------------

    if (
        hotspot_risk == "HIGH"
        and forecast_risk_value == "HIGH"
    ):
        return "CRITICAL_REVIEW"

    if (
        recent_event_count > 0
        and recent_event_score >= 0.75
        and (
            hotspot_risk == "HIGH"
            or forecast_risk_value == "HIGH"
        )
    ):
        return "CRITICAL_REVIEW"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if hotspot_risk == "HIGH":
        return "HIGH"

    if (
        forecast_risk_value == "HIGH"
        and recent_event_score
        >= HIGH_EVENT_SUPPORT_SCORE
    ):
        return "HIGH"

    if (
        forecast_risk_value == "HIGH"
        and current_pm25 >= 52
    ):
        return "HIGH"

    if (
        recent_event_score >= 0.70
        and hotspot_score >= 0.60
    ):
        return "HIGH"

    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if hotspot_risk == "MODERATE":
        return "MODERATE"

    if forecast_risk_value == "MODERATE":
        return "MODERATE"

    if recent_event_score >= 0.50:
        return "MODERATE"

    if (
        current_pm25 >= 37.8
        and forecast_value >= current_pm25
    ):
        return "MODERATE"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# ALERT REASON
# ============================================================

def build_alert_reason(
    hotspot_risk,
    forecast_risk_value,
    current_pm25,
    forecast_value,
    recent_event_count,
    recent_source_context,
    recent_event_score,
):
    reasons = []

    if hotspot_risk == "HIGH":
        reasons.append(
            "high hotspot score"
        )

    elif hotspot_risk == "MODERATE":
        reasons.append(
            "moderate hotspot score"
        )

    if forecast_risk_value == "HIGH":
        reasons.append(
            "high 3-hour PM2.5 forecast"
        )

    elif forecast_risk_value == "MODERATE":
        reasons.append(
            "moderate 3-hour PM2.5 forecast"
        )

    if (
        current_pm25 >= 37.8
        and forecast_value > current_pm25
    ):
        reasons.append(
            "forecast indicates rising PM2.5"
        )

    if recent_event_count > 0:

        context_text = (
            recent_source_context
            .replace(
                "_",
                " ",
            )
            .lower()
        )

        reasons.append(
            f"recent satellite-linked event: "
            f"{context_text}"
        )

    if not reasons:
        return (
            "No immediate elevated risk signal detected"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(
    priority,
    source_context,
):
    if priority == "CRITICAL_REVIEW":

        return (
            "Immediate environmental review recommended; "
            "verify local conditions and coordinate relevant "
            "monitoring or response teams."
        )

    if (
        source_context
        == "POTENTIAL_AGRICULTURAL_BURNING"
    ):

        return (
            "Verify potential agricultural burning activity "
            "and monitor downwind air-quality conditions."
        )

    if (
        source_context
        == "POTENTIAL_REGIONAL_TRANSPORT"
    ):

        return (
            "Monitor downwind stations and coordinate "
            "regional verification."
        )

    if priority == "HIGH":

        return (
            "Prioritize verification and increase monitoring "
            "of the affected area."
        )

    if priority == "MODERATE":

        return (
            "Continue enhanced monitoring and review "
            "supporting environmental evidence."
        )

    return (
        "Continue routine monitoring."
    )


# ============================================================
# MAIN
# ============================================================

def build_authority_alerts():

    print("\n==============================")
    print("FINAL AUTHORITY ALERT ENGINE")
    print("==============================")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    forecast_data = load_forecast_data()

    hotspots = load_hotspots()

    event_data = load_event_intelligence()

    model, model_features = load_model()

    # --------------------------------------------------------
    # THRESHOLDS
    # --------------------------------------------------------

    q75, q90 = calculate_forecast_thresholds(
        forecast_data
    )

    print(
        f"PM2.5 moderate threshold: {q75:.2f}"
    )

    print(
        f"PM2.5 high threshold: {q90:.2f}"
    )

    # --------------------------------------------------------
    # LATEST FORECASTS
    # --------------------------------------------------------

    latest = generate_latest_forecasts(
        forecast_data,
        model,
        model_features,
    )

    print(
        "Stations with latest forecast inputs:",
        latest["station_id"].nunique(),
    )

    # --------------------------------------------------------
    # SNAPSHOT TIME
    # --------------------------------------------------------

    snapshot_time = latest[
        "timestamp"
    ].max()

    print(
        "Latest available data timestamp:",
        snapshot_time,
    )

    # --------------------------------------------------------
    # MERGE HOTSPOTS
    # --------------------------------------------------------

    alerts = latest.merge(
        hotspots,
        on="station_id",
        how="left",
    )

    alerts[
        "hotspot_score"
    ] = alerts[
        "hotspot_score"
    ].fillna(0.0)

    alerts[
        "risk_level"
    ] = alerts[
        "risk_level"
    ].fillna("LOW")

    # --------------------------------------------------------
    # BUILD EVENT CONTEXT
    # --------------------------------------------------------

    event_rows = []

    for _, row in alerts.iterrows():

        summary = (
            build_recent_event_summary(
                station_id=row[
                    "station_id"
                ],
                snapshot_time=snapshot_time,
                event_data=event_data,
            )
        )

        event_rows.append(
            summary
        )

    event_summary = pd.DataFrame(
        event_rows,
        index=alerts.index,
    )

    alerts = pd.concat(
        [
            alerts,
            event_summary,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # FORECAST RISK
    # --------------------------------------------------------

    alerts[
        "forecast_risk"
    ] = alerts[
        "predicted_pm25_3h"
    ].apply(
        lambda value:
        forecast_risk(
            value,
            q75,
            q90,
        )
    )

    # --------------------------------------------------------
    # CURRENT PM2.5
    # --------------------------------------------------------

    alerts[
        "current_pm25"
    ] = pd.to_numeric(
        alerts[
            "pm25"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    alerts[
        "alert_priority"
    ] = alerts.apply(
        lambda row:
        calculate_alert_priority(
            hotspot_risk=row[
                "risk_level"
            ],
            forecast_risk_value=row[
                "forecast_risk"
            ],
            hotspot_score=row[
                "hotspot_score"
            ],
            forecast_value=row[
                "predicted_pm25_3h"
            ],
            current_pm25=row[
                "current_pm25"
            ],
            recent_event_score=row[
                "recent_event_score"
            ],
            recent_event_count=row[
                "recent_event_count"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    alerts[
        "alert_reason"
    ] = alerts.apply(
        lambda row:
        build_alert_reason(
            hotspot_risk=row[
                "risk_level"
            ],
            forecast_risk_value=row[
                "forecast_risk"
            ],
            current_pm25=row[
                "current_pm25"
            ],
            forecast_value=row[
                "predicted_pm25_3h"
            ],
            recent_event_count=row[
                "recent_event_count"
            ],
            recent_source_context=row[
                "recent_source_context"
            ],
            recent_event_score=row[
                "recent_event_score"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    alerts[
        "recommended_action"
    ] = alerts.apply(
        lambda row:
        recommended_action(
            priority=row[
                "alert_priority"
            ],
            source_context=row[
                "recent_source_context"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # DATA FRESHNESS
    # --------------------------------------------------------

    alerts[
        "data_status"
    ] = (
        "Latest available synchronized dataset"
    )

    # --------------------------------------------------------
    # SNAPSHOT TYPE
    # --------------------------------------------------------

    alerts[
        "snapshot_type"
    ] = "MODEL_REPLAY"

    # --------------------------------------------------------
    # ALERT ID
    # --------------------------------------------------------

    alerts.insert(
        0,
        "alert_id",
        [
            f"ALT-{index + 1:04d}"
            for index
            in range(
                len(alerts)
            )
        ],
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    priority_order = {
        "CRITICAL_REVIEW": 0,
        "HIGH": 1,
        "MODERATE": 2,
        "LOW": 3,
    }

    alerts[
        "_priority_order"
    ] = alerts[
        "alert_priority"
    ].map(
        priority_order
    )

    alerts = (
        alerts
        .sort_values(
            [
                "_priority_order",
                "predicted_pm25_3h",
                "hotspot_score",
                "recent_event_score",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "_priority_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # ROUND
    # --------------------------------------------------------

    numeric_columns = [
        "latitude",
        "longitude",
        "pm25",
        "pm10",
        "no2",
        "predicted_pm25_3h",
        "hotspot_score",
        "recent_event_score",
        "recent_event_count",
    ]

    for column in numeric_columns:

        if column in alerts.columns:

            alerts[column] = pd.to_numeric(
                alerts[column],
                errors="coerce",
            )

    round_columns = [
        "latitude",
        "longitude",
        "pm25",
        "pm10",
        "no2",
        "predicted_pm25_3h",
        "hotspot_score",
        "recent_event_score",
    ]

    for column in round_columns:

        if column in alerts.columns:

            alerts[column] = (
                alerts[column]
                .round(2)
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("AUTHORITY ALERT RESULTS")
    print("==============================")

    print(
        "Stations evaluated:",
        len(alerts),
    )

    print(
        "\nAlert priority:"
    )

    print(
        alerts[
            "alert_priority"
        ]
        .value_counts()
        .reindex(
            [
                "CRITICAL_REVIEW",
                "HIGH",
                "MODERATE",
                "LOW",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nForecast risk:"
    )

    print(
        alerts[
            "forecast_risk"
        ]
        .value_counts()
        .reindex(
            [
                "HIGH",
                "MODERATE",
                "LOW",
                "UNKNOWN",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nStations with recent satellite-linked evidence:",
        int(
            (
                alerts[
                    "recent_event_count"
                ]
                > 0
            ).sum()
        ),
    )

    print(
        "\nTop 20 authority alerts:"
    )

    display_columns = [
        "alert_id",
        "station",
        "alert_priority",
        "pm25",
        "predicted_pm25_3h",
        "forecast_risk",
        "hotspot_score",
        "recent_event_count",
        "recent_source_context",
        "recent_event_score",
        "alert_reason",
        "recommended_action",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in alerts.columns
    ]

    print(
        alerts[
            available_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nFinal authority alert engine completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    build_authority_alerts()