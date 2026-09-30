import json
from pathlib import Path

import pandas as pd


RAW_FILE = Path("data/raw/india_locations.json")
OUTPUT_FILE = Path("data/processed/india_sensor_inventory.csv")

REQUIRED_POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
}


def build_sensor_inventory():
    """Build a complete sensor inventory from the saved OpenAQ location data."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RAW_FILE}"
        )

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    locations = data.get("results", [])

    sensor_records = []

    for location in locations:

        station_id = location.get("id")
        station_name = location.get("name")
        locality = location.get("locality")

        coordinates = location.get("coordinates") or {}
        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")

        sensors = location.get("sensors") or []

        for sensor in sensors:

            parameter = sensor.get("parameter") or {}

            pollutant = parameter.get("name")
            units = parameter.get("units")
            display_name = parameter.get("displayName")

            # Keep only our six target pollutants
            if pollutant not in REQUIRED_POLLUTANTS:
                continue

            datetime_first = (
                sensor.get("datetimeFirst") or {}
            ).get("utc")

            datetime_last = (
                sensor.get("datetimeLast") or {}
            ).get("utc")

            latest = sensor.get("latest") or {}

            latest_datetime = (
                latest.get("datetime") or {}
            ).get("utc")

            latest_value = latest.get("value")

            sensor_records.append(
                {
                    "station_id": station_id,
                    "station": station_name,
                    "locality": locality,
                    "latitude": latitude,
                    "longitude": longitude,
                    "sensor_id": sensor.get("id"),
                    "pollutant": pollutant,
                    "display_name": display_name,
                    "units": units,
                    "datetime_first": datetime_first,
                    "datetime_last": datetime_last,
                    "latest_datetime": latest_datetime,
                    "latest_value": latest_value,
                }
            )

    sensor_df = pd.DataFrame(sensor_records)

    sensor_df["datetime_first"] = pd.to_datetime(
        sensor_df["datetime_first"],
        utc=True,
        errors="coerce",
    )

    sensor_df["datetime_last"] = pd.to_datetime(
        sensor_df["datetime_last"],
        utc=True,
        errors="coerce",
    )

    sensor_df["latest_datetime"] = pd.to_datetime(
        sensor_df["latest_datetime"],
        utc=True,
        errors="coerce",
    )

    sensor_df = sensor_df.sort_values(
        ["station_id", "pollutant", "datetime_last"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sensor_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Locations processed:", len(locations))
    print("Target sensor records:", len(sensor_df))
    print("Unique stations:", sensor_df["station_id"].nunique())
    print("Unique sensors:", sensor_df["sensor_id"].nunique())

    print("\nSensors by pollutant:")
    print(
        sensor_df["pollutant"]
        .value_counts()
        .sort_index()
    )

    print("\nSensors by unit:")
    print(
        sensor_df.groupby(
            ["pollutant", "units"]
        ).size()
    )

    print("\nSensor inventory saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    build_sensor_inventory()