from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/training_sensor_inventory.csv"
)

SENSOR_OUTPUT = Path(
    "data/processed/selected_training_sensors.csv"
)

STATION_OUTPUT = Path(
    "data/processed/selected_training_stations.csv"
)

REQUIRED_POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
}

MIN_OVERLAP_DAYS = 180


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["datetime_first"] = pd.to_datetime(
        df["datetime_first"],
        utc=True,
        errors="coerce"
    )

    df["datetime_last"] = pd.to_datetime(
        df["datetime_last"],
        utc=True,
        errors="coerce"
    )

    # Remove records without valid coverage dates.
    df = df.dropna(
        subset=[
            "datetime_first",
            "datetime_last"
        ]
    ).copy()

    # ---------------------------------------------------------
    # 1. Select the most recent sensor for each
    #    station + pollutant.
    # ---------------------------------------------------------

    current = (
        df.sort_values(
            [
                "station_id",
                "pollutant",
                "datetime_last",
                "datetime_first"
            ],
            ascending=[
                True,
                True,
                False,
                False
            ]
        )
        .groupby(
            [
                "station_id",
                "pollutant"
            ],
            as_index=False
        )
        .first()
    )

    print(
        "Current sensor records:",
        len(current)
    )

    # ---------------------------------------------------------
    # 2. Find stations that have all six pollutants.
    # ---------------------------------------------------------

    station_overlaps = []

    for station_id, group in current.groupby(
        "station_id"
    ):

        pollutants = set(
            group["pollutant"]
        )

        if not REQUIRED_POLLUTANTS.issubset(
            pollutants
        ):
            continue

        overlap_start = group[
            "datetime_first"
        ].max()

        overlap_end = group[
            "datetime_last"
        ].min()

        overlap_days = (
            overlap_end - overlap_start
        ).total_seconds() / 86400

        station_info = group.iloc[0]

        station_overlaps.append(
            {
                "station_id": station_id,
                "station": station_info["station"],
                "locality": station_info["locality"],
                "latitude": station_info["latitude"],
                "longitude": station_info["longitude"],
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_days": overlap_days,
            }
        )

    overlap_df = pd.DataFrame(
        station_overlaps
    )

    # ---------------------------------------------------------
    # 3. Keep stations with at least 180 days.
    # ---------------------------------------------------------

    selected_stations = (
        overlap_df[
            overlap_df["overlap_days"]
            >= MIN_OVERLAP_DAYS
        ]
        .sort_values(
            "overlap_days",
            ascending=False
        )
        .reset_index(drop=True)
    )

    selected_station_ids = set(
        selected_stations["station_id"]
    )

    # ---------------------------------------------------------
    # 4. Keep the six current sensors belonging
    #    to those stations.
    # ---------------------------------------------------------

    selected_sensors = current[
        current["station_id"].isin(
            selected_station_ids
        )
    ].copy()

    selected_sensors = selected_sensors.sort_values(
        [
            "station_id",
            "pollutant"
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Check that every selected station has six sensors.
    # ---------------------------------------------------------

    sensor_counts = (
        selected_sensors
        .groupby("station_id")["pollutant"]
        .nunique()
    )

    incomplete = sensor_counts[
        sensor_counts != 6
    ]

    if not incomplete.empty:
        raise ValueError(
            "Some selected stations do not have all six pollutants."
        )

    # ---------------------------------------------------------
    # 6. Save files.
    # ---------------------------------------------------------

    SENSOR_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_stations.to_csv(
        STATION_OUTPUT,
        index=False
    )

    selected_sensors.to_csv(
        SENSOR_OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Print summary.
    # ---------------------------------------------------------

    print("\n==============================")
    print("FINAL TRAINING SELECTION")
    print("==============================")

    print(
        "Selected stations:",
        len(selected_stations)
    )

    print(
        "Selected sensors:",
        len(selected_sensors)
    )

    print(
        "Expected sensors:",
        len(selected_stations) * 6
    )

    print(
        "Minimum overlap days:",
        round(
            selected_stations["overlap_days"].min(),
            1
        )
    )

    print(
        "Maximum overlap days:",
        round(
            selected_stations["overlap_days"].max(),
            1
        )
    )

    print(
        "\nPollutants:"
    )

    print(
        selected_sensors[
            "pollutant"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nUnits:"
    )

    print(
        selected_sensors[
            [
                "pollutant",
                "units"
            ]
        ]
        .drop_duplicates()
        .sort_values("pollutant")
        .to_string(index=False)
    )

    print(
        "\nSensor selection saved to:"
    )

    print(SENSOR_OUTPUT)

    print(
        "\nStation selection saved to:"
    )

    print(STATION_OUTPUT)


if __name__ == "__main__":
    main()