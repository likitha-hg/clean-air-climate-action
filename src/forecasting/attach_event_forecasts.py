from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_FILE = (
    PROJECT_ROOT
    / "data/processed/pollution_event_intelligence.csv"
)

FORECAST_DATASET_FILE = (
    PROJECT_ROOT
    / "data/processed/pm25_forecast_dataset.csv"
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
    / "pollution_event_intelligence_with_forecast.csv"
)


# ============================================================
# FORECAST SETTINGS
# ============================================================

FORECAST_HORIZON_HOURS = 3


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    model_bundle = joblib.load(
        MODEL_FILE
    )

    if not isinstance(
        model_bundle,
        dict,
    ):

        raise TypeError(
            "Expected model bundle dictionary."
        )

    if "model" not in model_bundle:

        raise KeyError(
            "Saved model bundle does not contain 'model'."
        )

    if "features" not in model_bundle:

        raise KeyError(
            "Saved model bundle does not contain 'features'."
        )

    model = model_bundle["model"]

    features = model_bundle["features"]

    return model, features


# ============================================================
# LOAD EVENTS
# ============================================================

def load_events():

    if not EVENT_FILE.exists():

        raise FileNotFoundError(
            f"Event intelligence file not found: {EVENT_FILE}"
        )

    events = pd.read_csv(
        EVENT_FILE
    )

    required_columns = [
        "correlation_id",
        "fire_event_id",
        "station_id",
        "station",
        "station_timestamp",
        "pm25_at_response",
        "event_score",
        "response_priority",
        "source_context",
    ]

    missing = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing:

        raise ValueError(
            f"Event intelligence is missing columns: {missing}"
        )

    events["station_timestamp"] = pd.to_datetime(
        events["station_timestamp"],
        errors="coerce",
        utc=True,
    )

    return events


# ============================================================
# LOAD FORECAST DATASET
# ============================================================

def load_forecast_dataset():

    if not FORECAST_DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Forecast dataset not found: "
            f"{FORECAST_DATASET_FILE}"
        )

    df = pd.read_csv(
        FORECAST_DATASET_FILE,
        parse_dates=[
            "timestamp"
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    if "station_id" not in df.columns:

        raise ValueError(
            "Forecast dataset has no station_id column."
        )

    return df


# ============================================================
# GENERATE HISTORICAL MODEL PREDICTIONS
# ============================================================

def generate_historical_predictions(
    forecast_df,
    model,
    feature_columns,
):
    """
    Generate the same 3-hour model prediction used by the
    operational forecasting system.

    Each prediction issued at timestamp T targets approximately:

        T + 3 hours
    """

    print(
        "\nGenerating historical 3-hour model predictions..."
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in forecast_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    valid = forecast_df.dropna(
        subset=feature_columns
    ).copy()

    print(
        "Rows with complete model features:",
        len(valid),
    )

    if valid.empty:

        raise RuntimeError(
            "No rows have complete model features."
        )

    X = valid[
        feature_columns
    ]

    predictions = model.predict(
        X
    )

    valid[
        "predicted_pm25_3h"
    ] = np.maximum(
        predictions,
        0,
    )

    valid[
        "forecast_target_timestamp"
    ] = (
        valid[
            "timestamp"
        ]
        + pd.Timedelta(
            hours=FORECAST_HORIZON_HOURS
        )
    )

    # Keep only fields needed for event matching.
    prediction_columns = [
        "station_id",
        "timestamp",
        "forecast_target_timestamp",
        "predicted_pm25_3h",
    ]

    if "target_pm25_3h" in valid.columns:

        prediction_columns.append(
            "target_pm25_3h"
        )

    prediction_data = (
        valid[
            prediction_columns
        ]
        .sort_values(
            [
                "station_id",
                "forecast_target_timestamp",
            ]
        )
        .drop_duplicates(
            subset=[
                "station_id",
                "forecast_target_timestamp",
            ],
            keep="last",
        )
    )

    return prediction_data


# ============================================================
# FORECAST RISK
# ============================================================

def classify_forecast_risk(
    value,
    q75,
    q90,
):
    """
    Use the same distribution-style risk thresholds as the
    operational dashboard.
    """

    if pd.isna(value):

        return "UNKNOWN"

    if value >= q90:

        return "HIGH"

    if value >= q75:

        return "MODERATE"

    return "LOW"


# ============================================================
# FORECAST EVIDENCE SCORE
# ============================================================

def calculate_forecast_evidence(
    forecast_value,
    current_pm25,
    q75,
    q90,
):
    """
    Convert forecast information into a normalized evidence
    score for the event engine.

    This is not a regulatory AQI score.
    """

    if pd.isna(
        forecast_value
    ):

        return 0.0

    forecast_value = float(
        forecast_value
    )

    if forecast_value >= q90:

        base_score = 1.0

    elif forecast_value >= q75:

        base_score = 0.70

    else:

        # Scale lower forecasts into a modest evidence range.
        denominator = max(
            q75,
            1.0,
        )

        base_score = min(
            0.50,
            forecast_value
            / denominator
            * 0.50,
        )

    # --------------------------------------------------------
    # FORECAST INCREASE
    # --------------------------------------------------------

    if (
        current_pm25 is not None
        and not pd.isna(current_pm25)
    ):

        current_pm25 = float(
            current_pm25
        )

        increase = (
            forecast_value
            - current_pm25
        )

        if increase >= 10:

            base_score += 0.15

        elif increase >= 5:

            base_score += 0.08

    return float(
        np.clip(
            base_score,
            0.0,
            1.0,
        )
    )


# ============================================================
# FINAL RESPONSE PRIORITY
# ============================================================

def calculate_integrated_priority(
    source_context,
    event_score,
    forecast_risk,
):
    """
    Combine event evidence with forecast evidence.

    The resulting priority is an operational prototype
    prioritization, not an official regulatory classification.
    """

    event_score = float(
        event_score
    )

    # --------------------------------------------------------
    # REGIONAL TRANSPORT
    # --------------------------------------------------------

    if source_context == (
        "POTENTIAL_REGIONAL_TRANSPORT"
    ):

        if (
            event_score >= 0.55
            and forecast_risk == "HIGH"
        ):

            return "CRITICAL_REVIEW"

        if event_score >= 0.60:

            return "HIGH"

        if (
            event_score >= 0.50
            or forecast_risk == "MODERATE"
        ):

            return "MODERATE"

        return "LOW"

    # --------------------------------------------------------
    # AGRICULTURAL BURNING
    # --------------------------------------------------------

    if source_context == (
        "POTENTIAL_AGRICULTURAL_BURNING"
    ):

        if (
            event_score >= 0.60
            and forecast_risk in [
                "HIGH",
                "MODERATE",
            ]
        ):

            return "HIGH"

        if event_score >= 0.60:

            return "HIGH"

        if (
            event_score >= 0.50
            or forecast_risk == "MODERATE"
        ):

            return "MODERATE"

        return "LOW"

    # --------------------------------------------------------
    # OTHER EVENTS
    # --------------------------------------------------------

    if (
        event_score >= 0.70
        and forecast_risk == "HIGH"
    ):

        return "HIGH"

    if (
        event_score >= 0.65
        or forecast_risk == "HIGH"
    ):

        return "HIGH"

    if (
        event_score >= 0.50
        or forecast_risk == "MODERATE"
    ):

        return "MODERATE"

    return "LOW"


# ============================================================
# MAIN
# ============================================================

def attach_forecasts():

    print("\n==============================")
    print("EVENT FORECAST INTEGRATION")
    print("==============================")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    events = load_events()

    forecast_df = load_forecast_dataset()

    model, feature_columns = load_model()

    print(
        "Event records:",
        len(events),
    )

    print(
        "Forecast dataset rows:",
        len(forecast_df),
    )

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    predictions = generate_historical_predictions(
        forecast_df,
        model,
        feature_columns,
    )

    print(
        "Historical prediction rows:",
        len(predictions),
    )

    # --------------------------------------------------------
    # HISTORICAL REFERENCE LEVELS
    # --------------------------------------------------------

    historical_pm25 = (
        forecast_df[
            "pm25"
        ]
        .dropna()
    )

    q75 = historical_pm25.quantile(
        0.75
    )

    q90 = historical_pm25.quantile(
        0.90
    )

    print(
        f"PM2.5 75th percentile: {q75:.2f}"
    )

    print(
        f"PM2.5 90th percentile: {q90:.2f}"
    )

    # --------------------------------------------------------
    # MATCH EVENT RESPONSE TIME TO FORECAST TARGET
    # --------------------------------------------------------

    events["station_timestamp"] = pd.to_datetime(
        events[
            "station_timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    predictions["forecast_target_timestamp"] = pd.to_datetime(
        predictions[
            "forecast_target_timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    integrated = events.merge(
        predictions,
        left_on=[
            "station_id",
            "station_timestamp",
        ],
        right_on=[
            "station_id",
            "forecast_target_timestamp",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # MATCH RATE
    # --------------------------------------------------------

    matched = (
        integrated[
            "predicted_pm25_3h"
        ]
        .notna()
    )

    print(
        "\nForecast matches:",
        int(matched.sum()),
        "/",
        len(integrated),
    )

    # --------------------------------------------------------
    # FORECAST RISK
    # --------------------------------------------------------

    integrated[
        "forecast_risk_at_response"
    ] = integrated[
        "predicted_pm25_3h"
    ].apply(
        lambda value:
        classify_forecast_risk(
            value,
            q75,
            q90,
        )
    )

    # --------------------------------------------------------
    # FORECAST EVIDENCE
    # --------------------------------------------------------

    integrated[
        "forecast_evidence_score"
    ] = integrated.apply(
        lambda row:
        calculate_forecast_evidence(
            forecast_value=row[
                "predicted_pm25_3h"
            ],
            current_pm25=row[
                "pm25_at_response"
            ],
            q75=q75,
            q90=q90,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # FORECAST CHANGE
    # --------------------------------------------------------

    integrated[
        "forecast_minus_response_pm25"
    ] = (
        integrated[
            "predicted_pm25_3h"
        ]
        - integrated[
            "pm25_at_response"
        ]
    )

    # --------------------------------------------------------
    # INTEGRATED SCORE
    # --------------------------------------------------------

    integrated[
        "integrated_event_score"
    ] = (
        0.75
        * integrated[
            "event_score"
        ]
        + 0.25
        * integrated[
            "forecast_evidence_score"
        ]
    )

    integrated[
        "integrated_event_score"
    ] = (
        integrated[
            "integrated_event_score"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # --------------------------------------------------------
    # INTEGRATED PRIORITY
    # --------------------------------------------------------

    integrated[
        "integrated_priority"
    ] = integrated.apply(
        lambda row:
        calculate_integrated_priority(
            source_context=row[
                "source_context"
            ],
            event_score=row[
                "integrated_event_score"
            ],
            forecast_risk=row[
                "forecast_risk_at_response"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # HUMAN-READABLE FORECAST EVIDENCE
    # --------------------------------------------------------

    def build_forecast_reason(row):

        if pd.isna(
            row["predicted_pm25_3h"]
        ):

            return (
                "No historical 3-hour forecast "
                "was available for the event response time."
            )

        predicted = float(
            row["predicted_pm25_3h"]
        )

        current = float(
            row["pm25_at_response"]
        )

        risk = row[
            "forecast_risk_at_response"
        ]

        change = (
            predicted
            - current
        )

        if change >= 10:

            trend = (
                "model predicted a substantial "
                "PM2.5 increase"
            )

        elif change >= 5:

            trend = (
                "model predicted a moderate "
                "PM2.5 increase"
            )

        elif change <= -5:

            trend = (
                "model predicted a PM2.5 decrease"
            )

        else:

            trend = (
                "model forecast was close to the "
                "observed PM2.5 level"
            )

        return (
            f"3-hour model forecast {predicted:.1f} µg/m³ "
            f"({risk} forecast risk); {trend}"
        )

    integrated[
        "forecast_evidence_summary"
    ] = integrated.apply(
        build_forecast_reason,
        axis=1,
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

    integrated[
        "_priority_order"
    ] = integrated[
        "integrated_priority"
    ].map(
        priority_order
    )

    integrated = (
        integrated
        .sort_values(
            [
                "_priority_order",
                "integrated_event_score",
            ],
            ascending=[
                True,
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
        "predicted_pm25_3h",
        "target_pm25_3h",
        "forecast_minus_response_pm25",
        "forecast_evidence_score",
        "integrated_event_score",
    ]

    for column in numeric_columns:

        if column in integrated.columns:

            integrated[column] = (
                pd.to_numeric(
                    integrated[column],
                    errors="coerce",
                )
                .round(3)
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrated.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("EVENT FORECAST RESULTS")
    print("==============================")

    print(
        "Total event records:",
        len(integrated),
    )

    print(
        "Forecast-linked records:",
        int(matched.sum()),
    )

    print(
        "Forecast match rate:",
        f"{matched.mean() * 100:.1f}%",
    )

    print(
        "\nForecast risk at event response:"
    )

    print(
        integrated[
            "forecast_risk_at_response"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nIntegrated priority:"
    )

    print(
        integrated[
            "integrated_priority"
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
        "\nSource context:"
    )

    print(
        integrated[
            "source_context"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # TOP EVENTS
    # --------------------------------------------------------

    print(
        "\nTop integrated events:"
    )

    display_columns = [
        "correlation_id",
        "fire_event_id",
        "station",
        "source_context",
        "integrated_priority",
        "fire_timestamp",
        "station_timestamp",
        "distance_km",
        "time_lag_hours",
        "pm25_at_response",
        "predicted_pm25_3h",
        "forecast_risk_at_response",
        "pm25_zscore",
        "cropland_fraction",
        "agricultural_burning_score",
        "event_score",
        "integrated_event_score",
        "evidence_level",
        "forecast_evidence_summary",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in integrated.columns
    ]

    print(
        integrated[
            available_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nEvent forecast integration completed successfully."
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    attach_forecasts()