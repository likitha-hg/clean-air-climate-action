from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/air_quality_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/pm25_forecast_dataset.csv"
)


POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


LAG_HOURS = [1, 3, 6]


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    print("Loading clean air-quality data...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "station_id",
            "timestamp",
            *POLLUTANTS
        ]
    ).copy()

    df = df.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    print(
        "Clean rows:",
        len(df)
    )

    # --------------------------------------------------
    # Time features
    # --------------------------------------------------

    local_time = df["timestamp"].dt.tz_convert(
        "Asia/Kolkata"
    )

    df["local_hour"] = local_time.dt.hour
    df["day_of_week"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month

    # Cyclic encoding for daily/weekly patterns
    df["hour_sin"] = np.sin(
        2 * np.pi * df["local_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["local_hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # --------------------------------------------------
    # Lag features
    #
    # We use timestamp-based merges instead of shift()
    # so a missing hour cannot accidentally become a
    # false "1-hour lag".
    # --------------------------------------------------

    base_columns = [
        "station_id",
        "timestamp",
    ]

    for lag_hours in LAG_HOURS:

        lag_data = df[
            base_columns + POLLUTANTS
        ].copy()

        lag_data["timestamp"] = (
            lag_data["timestamp"]
            + pd.Timedelta(hours=lag_hours)
        )

        rename_map = {
            pollutant:
                f"{pollutant}_lag_{lag_hours}h"
            for pollutant in POLLUTANTS
        }

        lag_data = lag_data.rename(
            columns=rename_map
        )

        df = df.merge(
            lag_data,
            on=[
                "station_id",
                "timestamp"
            ],
            how="left"
        )

    # --------------------------------------------------
    # 3-hour future PM2.5 target
    # --------------------------------------------------

    future = df[
        [
            "station_id",
            "timestamp",
            "pm25"
        ]
    ].copy()

    # Make future PM2.5 line up with the current row.
    future["timestamp"] = (
        future["timestamp"]
        - pd.Timedelta(hours=3)
    )

    future = future.rename(
        columns={
            "pm25": "target_pm25_3h"
        }
    )

    df = df.merge(
        future,
        on=[
            "station_id",
            "timestamp"
        ],
        how="left"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n==============================")
    print("FORECAST DATASET")
    print("==============================")

    print(
        "Total rows:",
        len(df)
    )

    print(
        "Rows with 3-hour target:",
        df["target_pm25_3h"].notna().sum()
    )

    print(
        "Rows without 3-hour target:",
        df["target_pm25_3h"].isna().sum()
    )

    print(
        "Stations:",
        df["station_id"].nunique()
    )

    print(
        "Features/columns:",
        len(df.columns)
    )

    print("\nTarget statistics:")

    print(
        df["target_pm25_3h"]
        .describe()
        .to_string()
    )

    print("\nSample rows:")

    print(
        df[
            [
                "station_id",
                "timestamp",
                "pm25",
                "pm25_lag_1h",
                "pm25_lag_3h",
                "pm25_lag_6h",
                "target_pm25_3h"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()