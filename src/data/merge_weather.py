from pathlib import Path

import pandas as pd


AIR_FILE = Path(
    "data/processed/air_quality_clean.csv"
)

WEATHER_FILE = Path(
    "data/raw/weather_training.csv"
)

OUTPUT_FILE = Path(
    "data/processed/air_quality_weather.csv"
)


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def main():

    print("Loading air-quality data...")

    air = pd.read_csv(AIR_FILE)

    print(
        "Air-quality rows:",
        len(air)
    )

    print("Loading weather data...")

    weather = pd.read_csv(WEATHER_FILE)

    print(
        "Weather rows:",
        len(weather)
    )

    # -----------------------------------------
    # Convert timestamps
    # -----------------------------------------

    air["timestamp"] = pd.to_datetime(
        air["timestamp"],
        utc=True,
        errors="coerce"
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
        errors="coerce"
    )

    # -----------------------------------------
    # Create matching hourly timestamp
    #
    # Example:
    # Air 13:30 -> Weather 13:00
    # -----------------------------------------

    air["weather_timestamp"] = (
        air["timestamp"].dt.floor("h")
    )

    # Keep only the columns we need
    weather_merge = weather[
        [
            "station_id",
            "timestamp",
            *WEATHER_COLUMNS,
        ]
    ].copy()

    weather_merge = weather_merge.rename(
        columns={
            "timestamp": "weather_timestamp"
        }
    )

    # -----------------------------------------
    # Remove duplicate weather records
    # -----------------------------------------

    weather_merge = (
        weather_merge
        .drop_duplicates(
            subset=[
                "station_id",
                "weather_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------
    # Merge
    # -----------------------------------------

    merged = air.merge(
        weather_merge,
        on=[
            "station_id",
            "weather_timestamp",
        ],
        how="left",
    )

    # -----------------------------------------
    # Remove helper column
    # -----------------------------------------

    merged = merged.drop(
        columns=["weather_timestamp"]
    )

    # -----------------------------------------
    # Sort
    # -----------------------------------------

    merged = merged.sort_values(
        [
            "station_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------
    # Save
    # -----------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print("\n==============================")
    print("AIR + WEATHER MERGE")
    print("==============================")

    print(
        "Merged rows:",
        len(merged)
    )

    print(
        "Stations:",
        merged["station_id"].nunique()
    )

    print(
        "Columns:",
        len(merged.columns)
    )

    print("\nWeather missing values:")

    print(
        merged[
            WEATHER_COLUMNS
        ]
        .isna()
        .sum()
        .to_string()
    )

    complete_weather = (
        merged[
            WEATHER_COLUMNS
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    print(
        "\nRows with all weather variables:",
        complete_weather
    )

    print(
        "Weather completeness:",
        round(
            complete_weather / len(merged) * 100,
            3
        ),
        "%"
    )

    print("\nSample merged rows:")

    print(
        merged[
            [
                "station_id",
                "timestamp",
                "pm25",
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()