from pathlib import Path
import time

import pandas as pd
import requests


AIR_FILE = Path(
    "data/processed/air_quality_clean.csv"
)

OUTPUT_FILE = Path(
    "data/raw/weather_training.csv"
)

URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "precipitation,"
    "surface_pressure,"
    "wind_speed_10m,"
    "wind_direction_10m"
)


def main():

    air = pd.read_csv(AIR_FILE)

    air["timestamp"] = pd.to_datetime(
        air["timestamp"],
        utc=True
    )

    # We only need stations that are actually
    # present in the clean ML dataset.
    stations = (
        air[
            [
                "station_id",
                "station",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates(
            subset=["station_id"]
        )
        .reset_index(drop=True)
    )

    start_date = (
        air["timestamp"]
        .min()
        .strftime("%Y-%m-%d")
    )

    end_date = (
        air["timestamp"]
        .max()
        .strftime("%Y-%m-%d")
    )

    print(
        "Weather period:",
        start_date,
        "→",
        end_date
    )

    print(
        "Stations:",
        len(stations)
    )

    all_weather = []

    for index, row in stations.iterrows():

        station_id = int(
            row["station_id"]
        )

        print(
            f"[{index + 1}/{len(stations)}] "
            f"{station_id} - {row['station']}"
        )

        params = {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": HOURLY_VARIABLES,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "GMT"
        }

        response = requests.get(
            URL,
            params=params,
            timeout=120
        )

        if response.status_code != 200:

            print(
                "    ERROR:",
                response.status_code
            )

            print(
                response.text[:500]
            )

            continue

        data = response.json()

        hourly = data.get(
            "hourly",
            {}
        )

        times = hourly.get(
            "time",
            []
        )

        if not times:

            print(
                "    No weather records returned."
            )

            continue

        weather_df = pd.DataFrame(
            {
                "station_id": station_id,
                "station": row["station"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "timestamp": pd.to_datetime(
                    times,
                    utc=True
                ),
                "temperature_2m": hourly.get(
                    "temperature_2m"
                ),
                "relative_humidity_2m": hourly.get(
                    "relative_humidity_2m"
                ),
                "precipitation": hourly.get(
                    "precipitation"
                ),
                "surface_pressure": hourly.get(
                    "surface_pressure"
                ),
                "wind_speed_10m": hourly.get(
                    "wind_speed_10m"
                ),
                "wind_direction_10m": hourly.get(
                    "wind_direction_10m"
                )
            }
        )

        all_weather.append(
            weather_df
        )

        print(
            "    Weather records:",
            len(weather_df)
        )

        # Small pause between stations.
        time.sleep(0.5)

    if not all_weather:
        raise RuntimeError(
            "No weather data was downloaded."
        )

    weather = pd.concat(
        all_weather,
        ignore_index=True
    )

    weather = weather.sort_values(
        [
            "station_id",
            "timestamp"
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    weather.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n==============================")
    print("WEATHER DOWNLOAD COMPLETE")
    print("==============================")

    print(
        "Stations:",
        weather["station_id"].nunique()
    )

    print(
        "Records:",
        len(weather)
    )

    print(
        "Earliest:",
        weather["timestamp"].min()
    )

    print(
        "Latest:",
        weather["timestamp"].max()
    )

    print(
        "Saved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()