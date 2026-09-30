from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/raw/openaq_hourly_training.csv"
)

OUTPUT_FILE = Path(
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
            f"Missing input file: {INPUT_FILE}"
        )

    print("Loading raw OpenAQ data...")

    df = pd.read_csv(INPUT_FILE)

    # -----------------------------------------
    # Convert timestamp
    # -----------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Remove records without valid timestamp
    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    print(
        "Raw observations:",
        len(df)
    )

    # -----------------------------------------
    # Keep only target pollutants
    # -----------------------------------------

    df = df[
        df["pollutant"].isin(POLLUTANTS)
    ].copy()

    # -----------------------------------------
    # Pivot long format → wide format
    # -----------------------------------------

    wide = (
        df.pivot_table(
            index=[
                "station_id",
                "station",
                "latitude",
                "longitude",
                "timestamp",
            ],
            columns="pollutant",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # Remove the pivot column name
    wide.columns.name = None

    # -----------------------------------------
    # Make sure all six pollutant columns exist
    # -----------------------------------------

    for pollutant in POLLUTANTS:

        if pollutant not in wide.columns:
            wide[pollutant] = pd.NA

    # -----------------------------------------
    # Keep only complete six-pollutant rows
    # -----------------------------------------

    complete = wide.dropna(
        subset=POLLUTANTS
    ).copy()

    # -----------------------------------------
    # Sort
    # -----------------------------------------

    complete = complete.sort_values(
        [
            "station_id",
            "timestamp"
        ]
    ).reset_index(drop=True)

    # -----------------------------------------
    # Final column order
    # -----------------------------------------

    complete = complete[
        [
            "station_id",
            "station",
            "latitude",
            "longitude",
            "timestamp",
            "pm25",
            "pm10",
            "no2",
            "so2",
            "co",
            "o3",
        ]
    ]

    # -----------------------------------------
    # Save
    # -----------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    complete.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print("\n==============================")
    print("ML DATASET CREATED")
    print("==============================")

    print(
        "Complete rows:",
        len(complete)
    )

    print(
        "Stations:",
        complete["station_id"].nunique()
    )

    print(
        "Earliest timestamp:",
        complete["timestamp"].min()
    )

    print(
        "Latest timestamp:",
        complete["timestamp"].max()
    )

    print("\nMissing values:")
    print(
        complete.isna()
        .sum()
        .to_string()
    )

    print("\nFirst 10 rows:")
    print(
        complete.head(10).to_string(
            index=False
        )
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()