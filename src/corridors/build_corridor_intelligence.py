from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MEMBERSHIP_FILE = Path(
    "data/processed/corridor_station_membership.csv"
)

ALERT_FILE = Path(
    "data/processed/authority_alerts_final.csv"
)

OUTPUT_FILE = Path(
    "data/processed/corridor_intelligence.csv"
)

STATION_OUTPUT_FILE = Path(
    "data/processed/corridor_station_signals.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CORRIDORS = [
    "DMIC",
    "AKIC",
    "CBIC",
    "VCIC",
    "ECEC",
    "BMIC",
]

# Historical thresholds already used by the authority alert engine.
# These are descriptive pressure indicators, not AQI categories.
PM25_ELEVATED_THRESHOLD = 37.80
PM25_HIGH_THRESHOLD = 52.00

# Signal-score weights.
WEIGHTS = {
    "current_pressure": 0.20,
    "forecast_pressure": 0.25,
    "hotspot_pressure": 0.20,
    "event_pressure": 0.15,
    "authority_pressure": 0.20,
}


# ============================================================
# LOAD DATA
# ============================================================

membership = pd.read_csv(MEMBERSHIP_FILE)
alerts = pd.read_csv(ALERT_FILE)


# ============================================================
# VALIDATION
# ============================================================

required_membership_columns = [
    "corridor_id",
    "corridor_name",
    "station_id",
    "station",
    "state",
]

required_alert_columns = [
    "station_id",
    "station",
    "pm25",
    "predicted_pm25_3h",
    "hotspot_score",
    "recent_event_count",
    "forecast_risk",
    "alert_priority",
]

missing_membership = [
    column
    for column in required_membership_columns
    if column not in membership.columns
]

missing_alert = [
    column
    for column in required_alert_columns
    if column not in alerts.columns
]

if missing_membership:
    raise ValueError(
        "Missing membership columns: "
        + ", ".join(missing_membership)
    )

if missing_alert:
    raise ValueError(
        "Missing alert columns: "
        + ", ".join(missing_alert)
    )


# ============================================================
# FILTER CORRIDORS
# ============================================================

membership = membership[
    membership["corridor_id"].isin(CORRIDORS)
].copy()


# ============================================================
# CLEAN ALERT DATA
# ============================================================

alerts = alerts[
    required_alert_columns
].copy()

alerts["pm25"] = pd.to_numeric(
    alerts["pm25"],
    errors="coerce",
)

alerts["predicted_pm25_3h"] = pd.to_numeric(
    alerts["predicted_pm25_3h"],
    errors="coerce",
)

alerts["hotspot_score"] = pd.to_numeric(
    alerts["hotspot_score"],
    errors="coerce",
)

alerts["recent_event_count"] = pd.to_numeric(
    alerts["recent_event_count"],
    errors="coerce",
).fillna(0)

# One latest authority row per station.
alerts = alerts.drop_duplicates(
    subset=["station_id"]
)


# ============================================================
# MERGE STATIONS WITH SIGNALS
# ============================================================

merged = membership.merge(
    alerts,
    on=["station_id", "station"],
    how="left",
)


# ============================================================
# BUILD STATION-LEVEL SIGNALS
# ============================================================

merged["current_pm25_elevated"] = (
    merged["pm25"]
    >= PM25_ELEVATED_THRESHOLD
)

merged["current_pm25_high"] = (
    merged["pm25"]
    >= PM25_HIGH_THRESHOLD
)

merged["forecast_pm25_elevated"] = (
    merged["predicted_pm25_3h"]
    >= PM25_ELEVATED_THRESHOLD
)

merged["forecast_pm25_high"] = (
    merged["predicted_pm25_3h"]
    >= PM25_HIGH_THRESHOLD
)

merged["forecast_pressure"] = merged[
    "forecast_risk"
].isin(
    [
        "MODERATE",
        "HIGH",
    ]
)

merged["authority_pressure"] = merged[
    "alert_priority"
].isin(
    [
        "MODERATE",
        "HIGH",
        "CRITICAL_REVIEW",
    ]
)

merged["event_signal"] = (
    merged["recent_event_count"] > 0
)

merged["hotspot_signal"] = (
    merged["hotspot_score"] > 0
)

merged["forecast_change"] = (
    merged["predicted_pm25_3h"]
    - merged["pm25"]
)


# ============================================================
# CORRIDOR AGGREGATION
# ============================================================

corridor_rows = []

for corridor_id in CORRIDORS:

    corridor = merged[
        merged["corridor_id"] == corridor_id
    ].copy()

    if corridor.empty:
        continue

    associated_stations = (
        corridor["station_id"]
        .nunique()
    )

    monitored = corridor[
        corridor["pm25"].notna()
    ]

    forecasted = corridor[
        corridor["predicted_pm25_3h"].notna()
    ]

    signal_stations = corridor[
        corridor[
            [
                "pm25",
                "predicted_pm25_3h",
                "hotspot_score",
                "recent_event_count",
                "forecast_risk",
                "alert_priority",
            ]
        ].notna().any(axis=1)
    ]

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    current_count = monitored[
        "station_id"
    ].nunique()

    forecast_count = forecasted[
        "station_id"
    ].nunique()

    forecast_coverage = (
        forecast_count / associated_stations
        if associated_stations
        else 0
    )

    current_coverage = (
        current_count / associated_stations
        if associated_stations
        else 0
    )

    # --------------------------------------------------------
    # PM2.5 PRESSURE
    # --------------------------------------------------------

    current_mean = (
        monitored["pm25"].mean()
        if not monitored.empty
        else None
    )

    forecast_mean = (
        forecasted["predicted_pm25_3h"].mean()
        if not forecasted.empty
        else None
    )

    forecast_change_mean = (
        forecasted["forecast_change"].mean()
        if not forecasted.empty
        else None
    )

    current_elevated_count = (
        monitored["current_pm25_elevated"].sum()
    )

    current_high_count = (
        monitored["current_pm25_high"].sum()
    )

    forecast_elevated_count = (
        forecasted["forecast_pm25_elevated"].sum()
    )

    forecast_high_count = (
        forecasted["forecast_pm25_high"].sum()
    )

    current_pressure_rate = (
        current_elevated_count / current_count
        if current_count
        else 0
    )

    forecast_pressure_rate = (
        forecast_elevated_count / forecast_count
        if forecast_count
        else 0
    )

    # --------------------------------------------------------
    # HOTSPOT PRESSURE
    # --------------------------------------------------------

    hotspot_scores = pd.to_numeric(
        corridor["hotspot_score"],
        errors="coerce",
    ).dropna()

    mean_hotspot_score = (
        hotspot_scores.mean()
        if not hotspot_scores.empty
        else 0
    )

    hotspot_station_count = corridor[
        corridor["hotspot_signal"]
    ]["station_id"].nunique()

    # --------------------------------------------------------
    # EVENT PRESSURE
    # --------------------------------------------------------

    event_station_count = corridor[
        corridor["event_signal"]
    ]["station_id"].nunique()

    event_signal_rate = (
        event_station_count / current_count
        if current_count
        else 0
    )

    recent_event_count = int(
        corridor["recent_event_count"]
        .fillna(0)
        .sum()
    )

    # --------------------------------------------------------
    # AUTHORITY PRESSURE
    # --------------------------------------------------------

    authority_station_count = corridor[
        corridor["authority_pressure"]
    ]["station_id"].nunique()

    authority_pressure_rate = (
        authority_station_count / current_count
        if current_count
        else 0
    )

    # --------------------------------------------------------
    # FORECAST RISK COUNTS
    # --------------------------------------------------------

    forecast_moderate_count = corridor[
        corridor["forecast_risk"] == "MODERATE"
    ]["station_id"].nunique()

    forecast_high_risk_count = corridor[
        corridor["forecast_risk"] == "HIGH"
    ]["station_id"].nunique()

    # --------------------------------------------------------
    # SIGNAL SCORE
    # --------------------------------------------------------

    current_component = (
        current_pressure_rate * 100
    )

    forecast_component = (
        forecast_pressure_rate * 100
    )

    hotspot_component = (
        mean_hotspot_score * 100
    )

    event_component = (
        event_signal_rate * 100
    )

    authority_component = (
        authority_pressure_rate * 100
    )

    signal_score = (
        current_component
        * WEIGHTS["current_pressure"]
        +
        forecast_component
        * WEIGHTS["forecast_pressure"]
        +
        hotspot_component
        * WEIGHTS["hotspot_pressure"]
        +
        event_component
        * WEIGHTS["event_pressure"]
        +
        authority_component
        * WEIGHTS["authority_pressure"]
    )

    signal_score = round(
        float(signal_score),
        2,
    )

    # --------------------------------------------------------
    # SIGNAL LEVEL
    # --------------------------------------------------------
    #
    # Coverage below 50% is explicitly surfaced as
    # LIMITED_COVERAGE rather than treating sparse data
    # as equivalent to strong monitoring coverage.
    #

    if forecast_coverage < 0.50:
        signal_level = "LIMITED_COVERAGE"

    elif signal_score >= 60:
        signal_level = "HIGH"

    elif signal_score >= 35:
        signal_level = "MODERATE"

    else:
        signal_level = "LOW"

    # --------------------------------------------------------
    # DATA CONFIDENCE
    # --------------------------------------------------------

    if forecast_coverage >= 0.80:
        coverage_label = "HIGH_COVERAGE"

    elif forecast_coverage >= 0.50:
        coverage_label = "MEDIUM_COVERAGE"

    else:
        coverage_label = "LIMITED_COVERAGE"

    # --------------------------------------------------------
    # ROW
    # --------------------------------------------------------

    corridor_rows.append(
        {
            "corridor_id": corridor_id,
            "corridor_name": corridor[
                "corridor_name"
            ].iloc[0],

            "associated_stations": associated_stations,

            "current_pm25_stations": current_count,
            "forecast_stations": forecast_count,

            "current_coverage_pct": round(
                current_coverage * 100,
                1,
            ),

            "forecast_coverage_pct": round(
                forecast_coverage * 100,
                1,
            ),

            "mean_current_pm25": round(
                current_mean,
                2,
            ) if current_mean is not None else None,

            "mean_forecast_pm25_3h": round(
                forecast_mean,
                2,
            ) if forecast_mean is not None else None,

            "mean_forecast_change": round(
                forecast_change_mean,
                2,
            ) if forecast_change_mean is not None else None,

            "current_elevated_count": int(
                current_elevated_count
            ),

            "current_high_count": int(
                current_high_count
            ),

            "forecast_elevated_count": int(
                forecast_elevated_count
            ),

            "forecast_high_count": int(
                forecast_high_count
            ),

            "forecast_moderate_count": int(
                forecast_moderate_count
            ),

            "forecast_high_risk_count": int(
                forecast_high_risk_count
            ),

            "hotspot_station_count": int(
                hotspot_station_count
            ),

            "mean_hotspot_score": round(
                float(mean_hotspot_score),
                3,
            ),

            "event_station_count": int(
                event_station_count
            ),

            "recent_event_count": recent_event_count,

            "authority_alert_station_count": int(
                authority_station_count
            ),

            "signal_score": signal_score,

            "signal_level": signal_level,

            "coverage_label": coverage_label,
        }
    )


# ============================================================
# SAVE CORRIDOR INTELLIGENCE
# ============================================================

corridor_df = pd.DataFrame(
    corridor_rows
)

corridor_df = corridor_df.sort_values(
    "signal_score",
    ascending=False,
)

corridor_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SAVE STATION-LEVEL CORRIDOR SIGNALS
# ============================================================

station_columns = [
    "corridor_id",
    "corridor_name",
    "station_id",
    "station",
    "state",
    "pm25",
    "predicted_pm25_3h",
    "forecast_change",
    "hotspot_score",
    "recent_event_count",
    "forecast_risk",
    "alert_priority",
    "current_pm25_elevated",
    "current_pm25_high",
    "forecast_pm25_elevated",
    "forecast_pm25_high",
    "event_signal",
    "hotspot_signal",
    "authority_pressure",
]

station_df = merged[
    station_columns
].copy()

station_df.to_csv(
    STATION_OUTPUT_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 110)
print("CORRIDOR INTELLIGENCE")
print("=" * 110)

for _, row in corridor_df.iterrows():

    print()
    print(
        f"{row['corridor_id']} — "
        f"{row['corridor_name']}"
    )

    print(
        f"  Forecast coverage       : "
        f"{row['forecast_coverage_pct']:.1f}%"
    )

    print(
        f"  Mean current PM2.5     : "
        f"{row['mean_current_pm25']}"
    )

    print(
        f"  Mean 3h forecast PM2.5 : "
        f"{row['mean_forecast_pm25_3h']}"
    )

    print(
        f"  Mean forecast change    : "
        f"{row['mean_forecast_change']}"
    )

    print(
        f"  Forecast elevated      : "
        f"{row['forecast_elevated_count']}"
    )

    print(
        f"  Hotspot stations       : "
        f"{row['hotspot_station_count']}"
    )

    print(
        f"  Recent event stations  : "
        f"{row['event_station_count']}"
    )

    print(
        f"  Authority alert stations: "
        f"{row['authority_alert_station_count']}"
    )

    print(
        f"  Signal score           : "
        f"{row['signal_score']:.2f}"
    )

    print(
        f"  Signal level           : "
        f"{row['signal_level']}"
    )

    print(
        f"  Coverage status        : "
        f"{row['coverage_label']}"
    )


print()
print("=" * 110)
print(
    f"Saved: {OUTPUT_FILE}"
)
print(
    f"Saved: {STATION_OUTPUT_FILE}"
)
print("=" * 110)