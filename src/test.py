import requests


LATITUDE = 31.321907
LONGITUDE = 75.578914

START_DATE = "2026-03-21"
END_DATE = "2026-03-23"


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "surface_pressure,"
        "wind_speed_10m,"
        "wind_direction_10m"
    ),
    "wind_speed_unit": "ms",
    "timezone": "GMT",
}

response = requests.get(
    url,
    params=params,
    timeout=60,
)

print("Status code:", response.status_code)

data = response.json()

print("Timezone:", data.get("timezone"))
print("Latitude:", data.get("latitude"))
print("Longitude:", data.get("longitude"))

if response.status_code == 200:

    hourly = data.get("hourly", {})

    print(
        "\nHourly records:",
        len(hourly.get("time", []))
    )

    print(
        "\nAvailable variables:"
    )

    print(
        list(hourly.keys())
    )

    print(
        "\nFirst 5 records:"
    )

    for i in range(
        min(5, len(hourly.get("time", [])))
    ):

        print(
            hourly["time"][i],
            "| Temp:",
            hourly["temperature_2m"][i],
            "| Humidity:",
            hourly["relative_humidity_2m"][i],
            "| Wind:",
            hourly["wind_speed_10m"][i],
            "| Direction:",
            hourly["wind_direction_10m"][i],
            "| Rain:",
            hourly["precipitation"][i]
        )