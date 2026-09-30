from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/training_sensor_inventory.csv"
)


REQUIRED_POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


def main():

    df = pd.read_csv(INPUT_FILE)

    df["datetime_first"] = pd.to_datetime(
        df["datetime_first"],
        utc=True,
        errors="coerce",
    )

    df["datetime_last"] = pd.to_datetime(
        df["datetime_last"],
        utc=True,
        errors="coerce",
    )

    # Choose the sensor with the most recent end date
    # for each station + pollutant.
    current = (
        df.sort_values(
            ["station_id", "pollutant",
             "datetime_last", "datetime_first"],
            ascending=[True, True, False, False]
        )
        .groupby(
            ["station_id", "pollutant"],
            as_index=False
        )
        .first()
    )

    print("Current sensor records:", len(current))
    print(
        "Stations represented:",
        current["station_id"].nunique()
    )

    print("\n==============================")
    print("CURRENT SENSOR UNIT COUNTS")
    print("==============================")

    unit_counts = (
        current
        .groupby(["pollutant", "units"])
        .size()
        .reset_index(name="stations")
        .sort_values(
            ["pollutant", "stations"],
            ascending=[True, False]
        )
    )

    print(unit_counts.to_string(index=False))

    print("\n==============================")
    print("CURRENT SENSOR OVERLAP")
    print("==============================")

    station_results = []

    for station_id, group in current.groupby(
        "station_id"
    ):

        pollutants = set(
            group["pollutant"]
        )

        if not set(REQUIRED_POLLUTANTS).issubset(
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

        station_name = group["station"].iloc[0]

        station_results.append(
            {
                "station_id": station_id,
                "station": station_name,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_days": round(
                    overlap_days, 1
                ),
            }
        )

    overlap_df = pd.DataFrame(
        station_results
    )

    if overlap_df.empty:
        print("No complete six-pollutant station groups found.")
        return

    overlap_df = overlap_df.sort_values(
        "overlap_days",
        ascending=False
    ).reset_index(drop=True)

    print(
        "\nStations with all six current sensors:",
        len(overlap_df)
    )

    print(
        "\nAt least 30 days:",
        (
            overlap_df["overlap_days"] >= 30
        ).sum()
    )

    print(
        "At least 60 days:",
        (
            overlap_df["overlap_days"] >= 60
        ).sum()
    )

    print(
        "At least 90 days:",
        (
            overlap_df["overlap_days"] >= 90
        ).sum()
    )

    print(
        "At least 180 days:",
        (
            overlap_df["overlap_days"] >= 180
        ).sum()
    )

    print("\nStation overlap:")
    print(
        overlap_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()