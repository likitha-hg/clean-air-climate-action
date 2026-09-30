import time
from pathlib import Path

import pandas as pd
import requests

from src.config import OPENAQ_API_KEY


INPUT_FILE = Path(
    "data/processed/training_station_candidates.csv"
)

OUTPUT_FILE = Path(
    "data/processed/training_sensor_inventory.csv"
)

BASE_URL = "https://api.openaq.org/v3"

REQUIRED_POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
}


def get_sensors_for_station(station_id):
    """Get sensor metadata for one OpenAQ station."""

    url = f"{BASE_URL}/locations/{station_id}/sensors"

    params = {
        "limit": 100,
        "page": 1,
    }

    response = requests.get(
        url,
        headers={"X-API-Key": OPENAQ_API_KEY},
        params=params,
        timeout=30,
    )

    return response


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    stations = pd.read_csv(INPUT_FILE)

    all_records = []

    print(
        f"Training stations to inspect: {len(stations)}"
    )

    for index, row in stations.iterrows():

        station_id = int(row["station_id"])
        station_name = row["station"]

        print(
            f"[{index + 1}/{len(stations)}] "
            f"{station_id} - {station_name}"
        )

        success = False

        for attempt in range(3):

            response = get_sensors_for_station(
                station_id
            )

            if response.status_code == 200:

                success = True

                results = response.json().get(
                    "results",
                    []
                )

                for sensor in results:

                    parameter = (
                        sensor.get("parameter") or {}
                    )

                    pollutant = parameter.get("name")

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

                    coverage = (
                        sensor.get("coverage") or {}
                    )

                    all_records.append(
                        {
                            "station_id": station_id,
                            "station": station_name,
                            "locality": row["locality"],
                            "latitude": row["latitude"],
                            "longitude": row["longitude"],
                            "sensor_id": sensor.get("id"),
                            "pollutant": pollutant,
                            "units": parameter.get("units"),
                            "datetime_first": datetime_first,
                            "datetime_last": datetime_last,
                            "latest_datetime": latest_datetime,
                            "latest_value": latest.get("value"),
                            "coverage_percent": coverage.get(
                                "percentCoverage"
                            ),
                        }
                    )

                print(
                    f"    Sensors found: {len(results)}"
                )

                break

            elif response.status_code == 429:

                reset = response.headers.get(
                    "x-ratelimit-reset"
                )

                try:
                    sleep_seconds = max(
                        float(reset or 65),
                        65
                    )
                except ValueError:
                    sleep_seconds = 65

                print(
                    f"    Rate limit reached. "
                    f"Pausing {sleep_seconds:.0f}s."
                )

                time.sleep(sleep_seconds)

            else:

                print(
                    f"    Error {response.status_code}"
                )

                if attempt < 2:
                    time.sleep(3)

        if not success:
            print(
                f"    Could not retrieve station {station_id}"
            )

        # Stay comfortably below the API rate limit.
        time.sleep(1.2)

        # Save progress after every station.
        progress_df = pd.DataFrame(all_records)

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        progress_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

    # Final summary
    sensor_df = pd.DataFrame(all_records)

    if sensor_df.empty:
        print("\nNo sensor records were retrieved.")
        return

    sensor_df["datetime_first"] = pd.to_datetime(
        sensor_df["datetime_first"],
        utc=True,
        errors="coerce"
    )

    sensor_df["datetime_last"] = pd.to_datetime(
        sensor_df["datetime_last"],
        utc=True,
        errors="coerce"
    )

    print("\n==============================")
    print("TRAINING SENSOR INVENTORY")
    print("==============================")

    print(
        "Sensor records:",
        len(sensor_df)
    )

    print(
        "Stations:",
        sensor_df["station_id"].nunique()
    )

    print(
        "Sensors:",
        sensor_df["sensor_id"].nunique()
    )

    print("\nSensors by pollutant:")

    print(
        sensor_df["pollutant"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()