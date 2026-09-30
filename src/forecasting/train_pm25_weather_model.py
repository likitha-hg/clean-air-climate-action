from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


INPUT_FILE = Path(
    "data/processed/pm25_weather_forecast_dataset.csv"
)

MODEL_FILE = Path(
    "models/pm25_xgb_weather_model.joblib"
)

RESULTS_FILE = Path(
    "models/pm25_weather_model_results.csv"
)


POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]

LAG_FEATURES = []

for pollutant in POLLUTANTS:
    for lag in [1, 3, 6]:
        LAG_FEATURES.append(
            f"{pollutant}_lag_{lag}h"
        )


FEATURES = (
    # Current air-quality values
    POLLUTANTS

    # Historical air-quality values
    + LAG_FEATURES

    # Location
    + [
        "latitude",
        "longitude",
    ]

    # Time
    + [
        "local_hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    # Weather
    + [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_sin",
        "wind_direction_cos",
    ]
)

TARGET = "target_pm25_3h"


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    print("Loading weather forecasting dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    required_columns = FEATURES + [TARGET]

    model_df = (
        df
        .dropna(subset=required_columns)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Rows before filtering:",
        len(df)
    )

    print(
        "Rows available for training:",
        len(model_df)
    )

    # -----------------------------------------
    # Chronological 80/20 split
    # -----------------------------------------

    split_time = model_df["timestamp"].quantile(
        0.80
    )

    train_df = model_df[
        model_df["timestamp"] <= split_time
    ].copy()

    test_df = model_df[
        model_df["timestamp"] > split_time
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\n==============================")
    print("TIME-BASED SPLIT")
    print("==============================")

    print(
        "Split timestamp:",
        split_time
    )

    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "Testing rows:",
        len(test_df)
    )

    print(
        "Training period:",
        train_df["timestamp"].min(),
        "→",
        train_df["timestamp"].max()
    )

    print(
        "Testing period:",
        test_df["timestamp"].min(),
        "→",
        test_df["timestamp"].max()
    )

    # -----------------------------------------
    # XGBoost
    # -----------------------------------------

    print("\n==============================")
    print("TRAINING MODEL V2")
    print("==============================")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
        eval_metric="rmse",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_test, y_test)
        ],
        verbose=False,
    )

    print("Model V2 training complete.")

    # -----------------------------------------
    # Predictions
    # -----------------------------------------

    predictions = model.predict(
        X_test
    )

    predictions = predictions.clip(
        min=0
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n==============================")
    print("MODEL V2 PERFORMANCE")
    print("==============================")

    print(
        f"MAE : {mae:.3f} µg/m³"
    )

    print(
        f"RMSE: {rmse:.3f} µg/m³"
    )

    print(
        f"R²  : {r2:.3f}"
    )

    # -----------------------------------------
    # Compare with Model V1
    # -----------------------------------------

    print("\n==============================")
    print("V1 → V2 COMPARISON")
    print("==============================")

    v1_mae = 6.252
    v1_rmse = 14.242
    v1_r2 = 0.442

    mae_change = (
        (mae - v1_mae)
        / v1_mae
        * 100
    )

    rmse_change = (
        (rmse - v1_rmse)
        / v1_rmse
        * 100
    )

    r2_change = (
        (r2 - v1_r2)
        / abs(v1_r2)
        * 100
    )

    print(
        f"V1 MAE : {v1_mae:.3f}"
    )

    print(
        f"V2 MAE : {mae:.3f}"
    )

    print(
        f"MAE change: {mae_change:.2f}%"
    )

    print(
        f"\nV1 RMSE: {v1_rmse:.3f}"
    )

    print(
        f"V2 RMSE: {rmse:.3f}"
    )

    print(
        f"RMSE change: {rmse_change:.2f}%"
    )

    print(
        f"\nV1 R²: {v1_r2:.3f}"
    )

    print(
        f"V2 R²: {r2:.3f}"
    )

    print(
        f"R² change: {r2_change:.2f}%"
    )

    # -----------------------------------------
    # Feature importance
    # -----------------------------------------

    importance_df = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n==============================")
    print("TOP 15 FEATURES")
    print("==============================")

    print(
        importance_df
        .head(15)
        .to_string(index=False)
    )

    # -----------------------------------------
    # Sample predictions
    # -----------------------------------------

    prediction_df = test_df[
        [
            "station_id",
            "station",
            "timestamp",
            "pm25",
            TARGET,
        ]
    ].copy()

    prediction_df[
        "predicted_pm25_3h"
    ] = predictions

    print("\n==============================")
    print("SAMPLE PREDICTIONS")
    print("==============================")

    print(
        prediction_df
        .head(20)
        .to_string(index=False)
    )

    # -----------------------------------------
    # Save model
    # -----------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_FILE
    )

    results = pd.DataFrame(
        {
            "model": [
                "V1_air_quality_only",
                "V2_air_quality_weather",
            ],
            "MAE": [
                v1_mae,
                mae,
            ],
            "RMSE": [
                v1_rmse,
                rmse,
            ],
            "R2": [
                v1_r2,
                r2,
            ],
        }
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        "\nModel saved to:",
        MODEL_FILE
    )

    print(
        "Results saved to:",
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()