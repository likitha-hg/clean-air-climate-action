import json
from pathlib import Path

import pandas as pd

from src.data.openaq_client import get_india_locations


# Project directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
}


def build_station_inventory():
    """Download India locations and create a station inventory."""

    data = get_india_locations(limit=1000)

    results = data.get("results", [])

    # Save original API response
    raw_file = RAW_DIR / "india_locations.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    stations = []

    for location in results:

        sensors = location.get("sensors") or []

        available_pollutants = set()

        for sensor in sensors:
            parameter = sensor.get("parameter") or {}
            pollutant = parameter.get("name")

            if pollutant in REQUIRED_POLLUTANTS:
                available_pollutants.add(pollutant)

        stations.append(
            {
                "station_id": location.get("id"),
                "station": location.get("name"),
                "locality": location.get("locality"),
                "latitude": (
                    location.get("coordinates") or {}
                ).get("latitude"),
                "longitude": (
                    location.get("coordinates") or {}
                ).get("longitude"),
                "datetime_first": (
                    location.get("datetimeFirst") or {}
                ).get("utc"),
                "datetime_last": (
                    location.get("datetimeLast") or {}
                ).get("utc"),
                "has_pm25": "pm25" in available_pollutants,
                "has_pm10": "pm10" in available_pollutants,
                "has_no2": "no2" in available_pollutants,
                "has_so2": "so2" in available_pollutants,
                "has_co": "co" in available_pollutants,
                "has_o3": "o3" in available_pollutants,
                "all_six_pollutants": (
                    REQUIRED_POLLUTANTS.issubset(available_pollutants)
                ),
            }
        )

    stations_df = pd.DataFrame(stations)

    output_file = PROCESSED_DIR / "india_station_inventory.csv"

    stations_df.to_csv(output_file, index=False)

    six_pollutant_count = stations_df["all_six_pollutants"].sum()

    print("India locations downloaded:", len(stations_df))
    print("Stations with all 6 pollutants:", six_pollutant_count)
    print("Raw data saved to:", raw_file)
    print("Station inventory saved to:", output_file)


if __name__ == "__main__":
    build_station_inventory()