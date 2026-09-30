from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/raw/openaq_hourly_training.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Minute values present in the dataset
    minute_counts = (
        df["timestamp"]
        .dt.minute
        .value_counts()
        .sort_index()
    )

    print("\n==============================")
    print("TIMESTAMP MINUTES")
    print("==============================")

    print(
        minute_counts.to_string()
    )

    # Check station + pollutant timestamp counts
    combinations = (
        df.groupby(
            [
                "station_id",
                "pollutant"
            ]
        )
        .agg(
            rows=("timestamp", "size"),
            unique_timestamps=("timestamp", "nunique"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max")
        )
        .reset_index()
    )

    print("\n==============================")
    print("TIMESTAMP UNIQUENESS")
    print("==============================")

    print(
        "Total station-pollutant combinations:",
        len(combinations)
    )

    print(
        "Combinations with duplicate timestamps:",
        (
            combinations["rows"]
            > combinations["unique_timestamps"]
        ).sum()
    )

    # Check how many timestamps have all six pollutants
    six_pollutant_counts = (
        df.groupby(
            [
                "station_id",
                "timestamp"
            ]
        )["pollutant"]
        .nunique()
    )

    print("\n==============================")
    print("SIX-POLLUTANT TIMESTAMPS")
    print("==============================")

    print(
        "Station-timestamp rows:",
        len(six_pollutant_counts)
    )

    print(
        "Timestamps containing all 6 pollutants:",
        (
            six_pollutant_counts == 6
        ).sum()
    )

    print(
        "Timestamps containing 5 pollutants:",
        (
            six_pollutant_counts == 5
        ).sum()
    )

    print(
        "Timestamps containing 4 pollutants:",
        (
            six_pollutant_counts == 4
        ).sum()
    )

    # Show examples of timestamp alignment
    print("\n==============================")
    print("SAMPLE TIMESTAMPS")
    print("==============================")

    sample = (
        df[
            [
                "station_id",
                "station",
                "pollutant",
                "timestamp"
            ]
        ]
        .sort_values(
            [
                "station_id",
                "timestamp",
                "pollutant"
            ]
        )
        .head(30)
    )

    print(
        sample.to_string(index=False)
    )


if __name__ == "__main__":
    main()