from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


INPUT_FILE = Path(
    "data/processed/pm25_forecast_dataset.csv"
)

TARGET = "target_pm25_3h"


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Use the same rows that have a valid target
    # and current PM2.5 value.
    df = df.dropna(
        subset=[
            "pm25",
            TARGET
        ]
    ).copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Same chronological 80/20 split used by
    # the XGBoost model.
    split_time = df["timestamp"].quantile(
        0.80
    )

    test_df = df[
        df["timestamp"] > split_time
    ].copy()

    y_true = test_df[TARGET]

    # Persistence baseline:
    # current PM2.5 = predicted PM2.5 after 3 hours
    baseline_predictions = test_df["pm25"]

    mae = mean_absolute_error(
        y_true,
        baseline_predictions
    )

    rmse = mean_squared_error(
        y_true,
        baseline_predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        baseline_predictions
    )

    print("\n==============================")
    print("BASELINE PERFORMANCE")
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

    print(
        "\nBaseline method:"
    )

    print(
        "Predicted PM2.5 at +3h = "
        "current PM2.5"
    )

    print("\nTest rows:", len(test_df))


if __name__ == "__main__":
    main()