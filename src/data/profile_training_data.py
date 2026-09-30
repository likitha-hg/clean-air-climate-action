from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/raw/openaq_hourly_training.csv"
)


def main():

    print("Loading training data...")

    df = pd.read_csv(INPUT_FILE)

    print("\n==============================")
    print("BASIC DATASET SUMMARY")
    print("==============================")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print(
        "Stations:",
        df["station_id"].nunique()
    )

    print(
        "Sensors:",
        df["sensor_id"].nunique()
    )

    print(
        "Pollutants:",
        df["pollutant"].nunique()
    )

    print(
        "\nPollutants found:"
    )

    print(
        sorted(
            df["pollutant"].dropna().unique()
        )
    )

    # -----------------------------------
    # Timestamp analysis
    # -----------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    print("\n==============================")
    print("TIME RANGE")
    print("==============================")

    print(
        "Earliest:",
        df["timestamp"].min()
    )

    print(
        "Latest:",
        df["timestamp"].max()
    )

    # -----------------------------------
    # Missing values
    # -----------------------------------

    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        print("No missing values found.")
    else:
        print(
            missing.to_string()
        )

    # -----------------------------------
    # Pollutant counts
    # -----------------------------------

    print("\n==============================")
    print("OBSERVATIONS BY POLLUTANT")
    print("==============================")

    pollutant_counts = (
        df.groupby("pollutant")
        .size()
        .sort_index()
    )

    print(
        pollutant_counts.to_string()
    )

    # -----------------------------------
    # Unit check
    # -----------------------------------

    print("\n==============================")
    print("UNITS")
    print("==============================")

    units = (
        df[
            [
                "pollutant",
                "units"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "pollutant",
                "units"
            ]
        )
    )

    print(
        units.to_string(
            index=False
        )
    )

    # -----------------------------------
    # Duplicate measurements
    # -----------------------------------

    print("\n==============================")
    print("DUPLICATES")
    print("==============================")

    duplicate_columns = [
        "station_id",
        "sensor_id",
        "pollutant",
        "timestamp"
    ]

    duplicate_count = (
        df.duplicated(
            subset=duplicate_columns
        )
        .sum()
    )

    print(
        "Duplicate rows:",
        duplicate_count
    )

    # -----------------------------------
    # Station + pollutant coverage
    # -----------------------------------

    print("\n==============================")
    print("STATION / POLLUTANT COVERAGE")
    print("==============================")

    coverage = (
        df.groupby(
            [
                "station_id",
                "station",
                "pollutant"
            ]
        )
        .agg(
            observations=(
                "value",
                "count"
            ),
            first_timestamp=(
                "timestamp",
                "min"
            ),
            last_timestamp=(
                "timestamp",
                "max"
            )
        )
        .reset_index()
    )

    print(
        "Station-pollutant combinations:",
        len(coverage)
    )

    print(
        "\nObservation statistics per "
        "station-pollutant:"
    )

    print(
        coverage["observations"]
        .describe()
        .to_string()
    )

    # -----------------------------------
    # Stations with all six pollutants
    # -----------------------------------

    print("\n==============================")
    print("SIX-POLLUTANT CHECK")
    print("==============================")

    six_count = (
        coverage
        .groupby("station_id")["pollutant"]
        .nunique()
    )

    complete_stations = (
        six_count[six_count == 6]
    )

    print(
        "Stations with all six pollutants:",
        len(complete_stations)
    )

    incomplete_stations = (
        six_count[six_count < 6]
    )

    print(
        "Stations missing at least one pollutant:",
        len(incomplete_stations)
    )

    # -----------------------------------
    # Show lowest-coverage combinations
    # -----------------------------------

    print("\n==============================")
    print("LOWEST COVERAGE COMBINATIONS")
    print("==============================")

    print(
        coverage
        .sort_values(
            "observations"
        )
        [
            [
                "station_id",
                "station",
                "pollutant",
                "observations",
                "first_timestamp",
                "last_timestamp"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()