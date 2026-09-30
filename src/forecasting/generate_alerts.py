from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = PROJECT_ROOT / "data/processed/pm25_forecast_dataset.csv"
HOTSPOT_FILE = PROJECT_ROOT / "data/processed/hotspots_latest.csv"
MODEL_FILE = PROJECT_ROOT / "models/pm25_xgb_model.joblib"

OUTPUT_FILE = PROJECT_ROOT / "data/processed/authority_alerts.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading forecasting dataset...")
df = pd.read_csv(
    FEATURES_FILE,
    parse_dates=["timestamp"]
)

print("Loading hotspot results...")
hotspots = pd.read_csv(HOTSPOT_FILE)

print("Loading PM2.5 forecasting model...")
model_bundle = joblib.load(MODEL_FILE)


# ============================================================
# UNPACK SAVED MODEL
# ============================================================

if not isinstance(model_bundle, dict):
    raise TypeError(
        "Expected the saved model file to contain a dictionary "
        "with 'model' and 'features'."
    )

if "model" not in model_bundle:
    raise KeyError(
        "Saved model bundle does not contain a 'model' key."
    )

if "features" not in model_bundle:
    raise KeyError(
        "Saved model bundle does not contain a 'features' key."
    )

model = model_bundle["model"]
feature_columns = model_bundle["features"]

print("Model loaded successfully.")
print("Number of model features:", len(feature_columns))


# ============================================================
# VALIDATE FEATURES
# ============================================================

target_column = "target_pm25_3h"

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features in forecasting dataset: "
        f"{missing_features}"
    )


# ============================================================
# GET LATEST AVAILABLE ROW FOR EACH STATION
# ============================================================

latest_rows = (
    df.sort_values("timestamp")
      .groupby("station_id", as_index=False)
      .tail(1)
      .copy()
)

print(
    "Stations before feature validation:",
    latest_rows["station_id"].nunique()
)


# ============================================================
# KEEP ONLY ROWS WITH ALL MODEL INPUTS
# ============================================================

latest_rows = latest_rows.dropna(
    subset=feature_columns
).copy()

print(
    "Stations available for forecasting:",
    latest_rows["station_id"].nunique()
)

if latest_rows.empty:
    raise ValueError(
        "No station has all required features available for forecasting."
    )


# ============================================================
# GENERATE 3-HOUR PM2.5 FORECAST
# ============================================================

X_latest = latest_rows[feature_columns]

latest_rows["forecast_pm25_3h"] = model.predict(X_latest)


# Prevent impossible negative predictions
latest_rows["forecast_pm25_3h"] = (
    latest_rows["forecast_pm25_3h"]
    .clip(lower=0)
)


# ============================================================
# HISTORICAL PM2.5 REFERENCE LEVELS
# ============================================================

historical_pm25 = df["pm25"].dropna()

if historical_pm25.empty:
    raise ValueError(
        "No historical PM2.5 values available."
    )

q75 = historical_pm25.quantile(0.75)
q90 = historical_pm25.quantile(0.90)

print(
    f"Historical PM2.5 75th percentile: {q75:.2f}"
)

print(
    f"Historical PM2.5 90th percentile: {q90:.2f}"
)


# ============================================================
# FORECAST RISK CLASSIFICATION
# ============================================================

def forecast_risk(value):
    """
    Classify forecast PM2.5 relative to the historical
    distribution used by this prototype.
    """

    if value >= q90:
        return "HIGH"

    if value >= q75:
        return "MODERATE"

    return "LOW"


latest_rows["forecast_risk"] = (
    latest_rows["forecast_pm25_3h"]
    .apply(forecast_risk)
)


# ============================================================
# SELECT HOTSPOT COLUMNS
# ============================================================

required_hotspot_columns = [
    "station_id",
    "hotspot_score",
    "risk_level",
    "hotspot_cluster",
]

missing_hotspot_columns = [
    column
    for column in required_hotspot_columns
    if column not in hotspots.columns
]

if missing_hotspot_columns:
    raise ValueError(
        f"Missing columns in hotspot file: "
        f"{missing_hotspot_columns}"
    )

hotspots_small = hotspots[
    required_hotspot_columns
].copy()


# ============================================================
# MERGE FORECAST + HOTSPOT INFORMATION
# ============================================================

alerts = latest_rows.merge(
    hotspots_small,
    on="station_id",
    how="left"
)


# ============================================================
# HANDLE STATIONS WITHOUT HOTSPOT SCORES
# ============================================================

alerts["hotspot_score"] = (
    alerts["hotspot_score"]
    .fillna(0)
)

alerts["risk_level"] = (
    alerts["risk_level"]
    .fillna("LOW")
)

alerts["hotspot_cluster"] = (
    alerts["hotspot_cluster"]
    .fillna(-1)
    .astype(int)
)


# ============================================================
# COMBINED ALERT PRIORITY
# ============================================================

def combined_priority(row):
    hotspot_risk = row["risk_level"]
    forecast_risk_value = row["forecast_risk"]

    if (
        hotspot_risk == "HIGH"
        or forecast_risk_value == "HIGH"
    ):
        return "HIGH"

    if (
        hotspot_risk == "MODERATE"
        or forecast_risk_value == "MODERATE"
    ):
        return "MODERATE"

    return "LOW"


alerts["alert_priority"] = alerts.apply(
    combined_priority,
    axis=1
)


# ============================================================
# ALERT REASON
# ============================================================

def alert_reason(row):

    reasons = []

    if row["forecast_risk"] == "HIGH":
        reasons.append(
            "High predicted PM2.5 in 3 hours"
        )

    elif row["forecast_risk"] == "MODERATE":
        reasons.append(
            "Moderate predicted PM2.5 in 3 hours"
        )

    if row["risk_level"] == "HIGH":
        reasons.append(
            "High hotspot score"
        )

    elif row["risk_level"] == "MODERATE":
        reasons.append(
            "Moderate hotspot score"
        )

    if not reasons:
        return "No immediate elevated risk detected"

    return " + ".join(reasons)


alerts["alert_reason"] = alerts.apply(
    alert_reason,
    axis=1
)


# ============================================================
# ALERT PRIORITY ORDER
# ============================================================

priority_order = {
    "HIGH": 0,
    "MODERATE": 1,
    "LOW": 2,
}

alerts["_priority_order"] = (
    alerts["alert_priority"]
    .map(priority_order)
)


# ============================================================
# SORT RESULTS
# ============================================================

alerts = (
    alerts.sort_values(
        by=[
            "_priority_order",
            "forecast_pm25_3h",
            "hotspot_score",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
    .drop(columns="_priority_order")
)


# ============================================================
# ROUND NUMERIC VALUES FOR OUTPUT
# ============================================================

numeric_columns = [
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "no2",
    "forecast_pm25_3h",
    "hotspot_score",
]

for column in numeric_columns:
    if column in alerts.columns:
        alerts[column] = alerts[column].round(2)


# ============================================================
# SAVE ALERT DATASET
# ============================================================

alerts.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n==============================")
print("AUTHORITY ALERT SNAPSHOT")
print("==============================")

print(
    "Latest timestamp:",
    alerts["timestamp"].max()
)

print(
    "Stations with forecasts:",
    alerts["station_id"].nunique()
)

print("\nForecast risk distribution:")

print(
    alerts["forecast_risk"]
    .value_counts()
    .reindex(
        ["HIGH", "MODERATE", "LOW"],
        fill_value=0
    )
    .to_string()
)

print("\nCombined alert priority:")

print(
    alerts["alert_priority"]
    .value_counts()
    .reindex(
        ["HIGH", "MODERATE", "LOW"],
        fill_value=0
    )
    .to_string()
)


# ============================================================
# DISPLAY TOP ALERTS
# ============================================================

print("\nTop 15 authority alerts:")

display_columns = [
    "station_id",
    "station",
    "latitude",
    "longitude",
    "pm25",
    "forecast_pm25_3h",
    "forecast_risk",
    "hotspot_score",
    "risk_level",
    "alert_priority",
    "alert_reason",
]

print(
    alerts[display_columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print("\nAlert generation completed successfully.")