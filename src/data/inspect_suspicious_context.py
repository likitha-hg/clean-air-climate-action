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

SCREENING_LIMITS = {
    "pm25": 1000,
    "pm10": 2000,
    "no2": 1000,
    "so2": 1000,
    "co": 10000,
    "o3": 500,
}


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Find suspicious rows
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

    suspicious = df[
        suspicious_mask
    ].copy()

    print(
        "Suspicious rows:",
        len(suspicious)
    )

    print("\n==============================")
    print("SUSPICIOUS EVENTS")
    print("==============================")

    print(
        suspicious[
            [
                "station_id",
                "station",
                "timestamp",
                *POLLUTANTS
            ]
        ]
        .sort_values(
            ["station_id", "timestamp"]
        )
        .to_string(index=False)
    )

    # -----------------------------------------
    # Context around each suspicious event
    # -----------------------------------------

    print("\n==============================")
    print("CONTEXT AROUND EVENTS")
    print("==============================")

    for _, event in suspicious.iterrows():

        station_id = event["station_id"]
        event_time = event["timestamp"]

        context = df[
            (df["station_id"] == station_id)
            &
            (
                df["timestamp"].between(
                    event_time - pd.Timedelta(hours=3),
                    event_time + pd.Timedelta(hours=3)
                )
            )
        ].copy()

        context = context.sort_values(
            "timestamp"
        )

        print("\n--------------------------------")
        print(
            f"Station: {event['station']}"
        )
        print(
            f"Event: {event_time}"
        )
        print("--------------------------------")

        print(
            context[
                [
                    "timestamp",
                    *POLLUTANTS
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()