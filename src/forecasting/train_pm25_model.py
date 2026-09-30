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
    "data/processed/pm25_forecast_dataset.csv"
)

MODEL_FILE = Path(
    "models/pm25_xgb_model.joblib"
)

RESULTS_FILE = Path(
    "models/pm25_model_results.csv"
)


FEATURES = [
    # Current pollutant observations
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",

    # Historical pollutant values
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",

    "pm10_lag_1h",
    "pm10_lag_3h",
    "pm10_lag_6h",

    "no2_lag_1h",
    "no2_lag_3h",
    "no2_lag_6h",

    "so2_lag_1h",
    "so2_lag_3h",
    "so2_lag_6h",

    "co_lag_1h",
    "co_lag_3h",
    "co_lag_6h",

    "o3_lag_1h",
    "o3_lag_3h",
    "o3_lag_6h",

    # Location
    "latitude",
    "longitude",

    # Time
    "local_hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

TARGET = "target_pm25_3h"


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    print("Loading forecasting dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # --------------------------------------------------
    # Keep only rows where all model features and target
    # are available.
    # --------------------------------------------------

    required_columns = FEATURES + [TARGET]

    model_df = df.dropna(
        subset=required_columns
    ).copy()

    model_df = model_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        "Rows before feature filtering:",
        len(df)
    )

    print(
        "Rows available for training:",
        len(model_df)
    )

    # --------------------------------------------------
    # Time-based train/test split
    # --------------------------------------------------

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

    # --------------------------------------------------
    # XGBoost model
    # --------------------------------------------------

    print("\n==============================")
    print("TRAINING XGBOOST")
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

    print("Training complete.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # Prevent negative predictions
    predictions = predictions.clip(
        min=0
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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
    print("MODEL PERFORMANCE")
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

    # --------------------------------------------------
    # Sample predictions
    # --------------------------------------------------

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
        prediction_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Save model results
    # --------------------------------------------------

    results = pd.DataFrame(
        {
            "metric": [
                "MAE",
                "RMSE",
                "R2",
            ],
            "value": [
                mae,
                rmse,
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
        "Metrics saved to:",
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()