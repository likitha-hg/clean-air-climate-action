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

# Conservative screening bounds.
# These are QA thresholds, NOT health standards.
SCREENING_LIMITS = {
    "pm25": 1000,   # µg/m³
    "pm10": 2000,   # µg/m³
    "no2": 1000,    # ppb
    "so2": 1000,    # ppb
    "co": 10000,    # ppb
    "o3": 500,      # µg/m³
}


def main():

    df = pd.read_csv(INPUT_FILE)

    print("\n==============================")
    print("SUSPICIOUS VALUE INSPECTION")
    print("==============================")

    for pollutant in POLLUTANTS:

        negative_count = (
            df[pollutant] < 0
        ).sum()

        high_count = (
            df[pollutant]
            > SCREENING_LIMITS[pollutant]
        ).sum()

        print(
            f"\n{pollutant.upper()}"
        )

        print(
            "  Negative values:",
            negative_count
        )

        print(
            f"  Values > {SCREENING_LIMITS[pollutant]}:",
            high_count
        )

        if high_count > 0:

            suspicious = (
                df[
                    df[pollutant]
                    > SCREENING_LIMITS[pollutant]
                ]
                [
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
                .head(20)
            )

            print(
                "\n  Highest suspicious values:"
            )

            print(
                suspicious.to_string(
                    index=False
                )
            )

    # -----------------------------------------
    # Count rows with any suspicious pollutant
    # -----------------------------------------

    suspicious_mask = pd.Series(
        False,
        index=df.index
    )

    for pollutant in POLLUTANTS:

        suspicious_mask |= (
            df[pollutant] < 0
        )

        suspicious_mask |= (
            df[pollutant]
            > SCREENING_LIMITS[pollutant]
        )

    print(
        "\n=============================="
    )

    print(
        "Rows containing at least one "
        "suspicious value:",
        suspicious_mask.sum()
    )

    print(
        "Percentage of dataset:",
        round(
            suspicious_mask.mean() * 100,
            4
        ),
        "%"
    )


if __name__ == "__main__":
    main()