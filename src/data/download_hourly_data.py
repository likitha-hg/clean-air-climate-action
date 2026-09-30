from pathlib import Path
import time

import pandas as pd
import requests

from src.config import OPENAQ_API_KEY


INPUT_FILE = Path(
    "data/processed/selected_training_sensors.csv"
)

OUTPUT_FILE = Path(
    "data/raw/openaq_hourly_training.csv"
)

BASE_URL = "https://api.openaq.org/v3"

MAX_DAYS = 180
PAGE_LIMIT = 1000
REQUEST_DELAY = 1.1


def get_station_window(station_sensors):
    """Calculate a 180-day window shared by all six pollutants."""

    overlap_start = station_sensors["datetime_first"].max()
    overlap_end = station_sensors["datetime_last"].min()

    # Use the latest 180 days available in the common overlap.
    start_by_duration = (
        overlap_end - pd.Timedelta(days=MAX_DAYS)
    )

    start = max(
        overlap_start,
        start_by_duration
    )

    return start, overlap_end


def download_sensor(sensor_id, start, end):

    url = (
        f"{BASE_URL}/sensors/"
        f"{int(sensor_id)}/hours"
    )

    all_results = []
    page = 1

    while True:

        params = {
            "datetime_from": start.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "datetime_to": end.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "limit": PAGE_LIMIT,
            "page": page,
        }

        response = requests.get(
            url,
            headers={
                "X-API-Key": OPENAQ_API_KEY
            },
            params=params,
            timeout=60,
        )

        if response.status_code == 429:

            print(
                f"      Rate limit on sensor {sensor_id}. "
                f"Pausing 65 seconds..."
            )

            time.sleep(65)
            continue

        if response.status_code != 200:

            print(
                f"      ERROR {response.status_code} "
                f"for sensor {sensor_id}"
            )

            return []

        data = response.json()

        results = data.get(
            "results",
            []
        )

        all_results.extend(results)

        # Stop when fewer than PAGE_LIMIT records
        # are returned.
        if len(results) < PAGE_LIMIT:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    return all_results


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INPUT_FILE}"
        )

    sensors = pd.read_csv(
        INPUT_FILE
    )

    sensors["datetime_first"] = pd.to_datetime(
        sensors["datetime_first"],
        utc=True
    )

    sensors["datetime_last"] = pd.to_datetime(
        sensors["datetime_last"],
        utc=True
    )

    # Start fresh
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_stations = sensors["station_id"].nunique()

    stations_completed = 0

    total_measurements = 0

    # Process one station at a time
    for station_id, station_df in sensors.groupby(
        "station_id"
    ):

        station_name = (
            station_df["station"]
            .iloc[0]
        )

        start, end = get_station_window(
            station_df
        )

        print(
            "\n--------------------------------"
        )
        print(
            f"Station {station_id}: {station_name}"
        )
        print(
            f"Window: {start} → {end}"
        )
        print(
            "Sensors:",
            len(station_df)
        )

        station_records = []

        for _, sensor in station_df.iterrows():

            sensor_id = sensor["sensor_id"]
            pollutant = sensor["pollutant"]
            units = sensor["units"]

            print(
                f"    {pollutant.upper()} "
                f"(sensor {sensor_id})"
            )

            results = download_sensor(
                sensor_id,
                start,
                end
            )

            print(
                f"      Records: {len(results)}"
            )

            for item in results:

                period = (
                    item.get("period") or {}
                )

                datetime_from = (
                    period.get(
                        "datetimeFrom"
                    ) or {}
                ).get("utc")

                datetime_to = (
                    period.get(
                        "datetimeTo"
                    ) or {}
                ).get("utc")

                coverage = (
                    item.get("coverage") or {}
                )

                station_records.append(
                    {
                        "station_id": station_id,
                        "station": station_name,
                        "latitude": sensor["latitude"],
                        "longitude": sensor["longitude"],
                        "sensor_id": sensor_id,
                        "pollutant": pollutant,
                        "units": units,
                        "value": item.get("value"),
                        "timestamp": datetime_from,
                        "datetime_to": datetime_to,
                        "coverage_percent": coverage.get(
                            "percentComplete"
                        ),
                    }
                )

            time.sleep(
                REQUEST_DELAY
            )

        if station_records:

            station_output = pd.DataFrame(
                station_records
            )

            write_header = not OUTPUT_FILE.exists()

            station_output.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=write_header,
                index=False
            )

            total_measurements += len(
                station_output
            )

        stations_completed += 1

        print(
            f"Completed "
            f"{stations_completed}/"
            f"{total_stations} stations"
        )

        print(
            f"Total observations saved: "
            f"{total_measurements}"
        )

    print(
        "\n================================"
    )
    print(
        "HOURLY DATA DOWNLOAD COMPLETE"
    )
    print(
        "================================"
    )

    print(
        "Stations processed:",
        stations_completed
    )

    print(
        "Total observations:",
        total_measurements
    )

    print(
        "Saved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()