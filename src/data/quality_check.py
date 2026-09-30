from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/air_quality_ml.csv"
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    print("\n==============================")
    print("ML DATA QUALITY CHECK")
    print("==============================")

    print("Rows:", len(df))
    print("Stations:", df["station_id"].nunique())

    # -----------------------------------------
    # Data types
    # -----------------------------------------

    print("\n==============================")
    print("DATA TYPES")
    print("==============================")

    print(
        df[
            ["pm25", "pm10", "no2",
             "so2", "co", "o3"]
        ]
        .dtypes
        .to_string()
    )

    # -----------------------------------------
    # Missing values
    # -----------------------------------------

    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")

    print(
        df[POLLUTANTS]
        .isna()
        .sum()
        .to_string()
    )

    # -----------------------------------------
    # Negative values
    # -----------------------------------------

    print("\n==============================")
    print("NEGATIVE VALUES")
    print("==============================")

    negative_counts = {}

    for pollutant in POLLUTANTS:
        negative_counts[pollutant] = (
            df[pollutant] < 0
        ).sum()

    print(
        pd.Series(
            negative_counts
        ).to_string()
    )

    # -----------------------------------------
    # Statistical summary
    # -----------------------------------------

    print("\n==============================")
    print("POLLUTANT STATISTICS")
    print("==============================")

    statistics = df[POLLUTANTS].describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99
        ]
    ).T

    print(
        statistics[
            [
                "count",
                "mean",
                "std",
                "min",
                "1%",
                "5%",
                "50%",
                "95%",
                "99%",
                "max"
            ]
        ].to_string()
    )

    # -----------------------------------------
    # Very large values
    # -----------------------------------------

    print("\n==============================")
    print("TOP 10 HIGHEST VALUES")
    print("==============================")

    for pollutant in POLLUTANTS:

        print(
            f"\n{pollutant.upper()}:"
        )

        print(
            df[
                [
                    "station_id",
                    "station",
                    "timestamp",
                    pollutant
                ]
            ]
            .sort_values(
                pollutant,
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

    # -----------------------------------------
    # Rows per station
    # -----------------------------------------

    print("\n==============================")
    print("ROWS PER STATION")
    print("==============================")

    station_counts = (
        df.groupby(
            ["station_id", "station"]
        )
        .size()
        .sort_values()
    )

    print(
        station_counts.to_string()
    )

    print("\n==============================")
    print("QUALITY CHECK COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()