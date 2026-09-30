from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


INPUT_FILE = Path(
    "data/processed/air_quality_weather.csv"
)

OUTPUT_FILE = Path(
    "data/processed/hotspots_latest.csv"
)


POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
]


def percentile_score(series):
    """
    Convert values into percentile scores from 0 to 1.
    Higher = more elevated relative to the snapshot.
    """
    return series.rank(
        pct=True,
        method="average"
    )


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # --------------------------------------------------
    # Find the latest timestamp with enough stations
    # --------------------------------------------------

    station_counts = (
        df.groupby("timestamp")["station_id"]
        .nunique()
        .sort_index()
    )

    valid_times = station_counts[
        station_counts >= 15
    ]

    if valid_times.empty:
        raise ValueError(
            "No timestamp has at least 15 stations."
        )

    snapshot_time = valid_times.index.max()

    snapshot = df[
        df["timestamp"] == snapshot_time
    ].copy()

    # --------------------------------------------------
    # Keep one row per station
    # --------------------------------------------------

    snapshot = (
        snapshot
        .drop_duplicates(
            subset=["station_id"]
        )
        .copy()
    )

    print("\n==============================")
    print("HOTSPOT SNAPSHOT")
    print("==============================")

    print(
        "Snapshot timestamp:",
        snapshot_time
    )

    print(
        "Stations in snapshot:",
        len(snapshot)
    )

    # --------------------------------------------------
    # Nationwide percentile signals
    # --------------------------------------------------

    snapshot["pm25_score"] = percentile_score(
        snapshot["pm25"]
    )

    snapshot["pm10_score"] = percentile_score(
        snapshot["pm10"]
    )

    snapshot["no2_score"] = percentile_score(
        snapshot["no2"]
    )

    # Lower wind = more stagnant conditions.
    # Convert wind speed into a reversed percentile.
    wind_percentile = percentile_score(
        snapshot["wind_speed_10m"]
    )

    snapshot["stagnation_score"] = (
        1 - wind_percentile
    )

    # --------------------------------------------------
    # Station-specific PM2.5 anomaly
    # --------------------------------------------------

    station_stats = (
        df.groupby("station_id")
        ["pm25"]
        .agg(
            station_median="median",
            station_mean="mean",
            station_std="std"
        )
        .reset_index()
    )

    snapshot = snapshot.merge(
        station_stats,
        on="station_id",
        how="left"
    )

    # Avoid division by zero for very stable stations.
    snapshot["pm25_std_safe"] = (
        snapshot["station_std"]
        .replace(0, np.nan)
        .fillna(1.0)
    )

    snapshot["pm25_anomaly"] = (
        (
            snapshot["pm25"]
            - snapshot["station_median"]
        )
        / snapshot["pm25_std_safe"]
    )

    # Convert anomaly to a bounded 0-1 score.
    snapshot["anomaly_score"] = (
        snapshot["pm25_anomaly"]
        .clip(lower=0, upper=5)
        / 5
    )

    # --------------------------------------------------
    # Composite hotspot score
    # --------------------------------------------------

    snapshot["hotspot_score"] = (
        0.40 * snapshot["pm25_score"]
        + 0.20 * snapshot["pm10_score"]
        + 0.15 * snapshot["no2_score"]
        + 0.10 * snapshot["stagnation_score"]
        + 0.15 * snapshot["anomaly_score"]
    )

    # --------------------------------------------------
    # Risk classification
    # --------------------------------------------------

    def classify(score):

        if score >= 0.80:
            return "HIGH"

        if score >= 0.60:
            return "MODERATE"

        return "LOW"

    snapshot["risk_level"] = (
        snapshot["hotspot_score"]
        .apply(classify)
    )

    # --------------------------------------------------
    # Geographic clustering of elevated locations
    # --------------------------------------------------

    elevated = snapshot[
        snapshot["hotspot_score"] >= 0.60
    ].copy()

    if len(elevated) >= 2:

        coordinates = np.radians(
            elevated[
                [
                    "latitude",
                    "longitude"
                ]
            ].values
        )

        # Approximately 200 km neighbourhood radius.
        eps_km = 200

        earth_radius_km = 6371.0

        clustering = DBSCAN(
            eps=eps_km / earth_radius_km,
            min_samples=2,
            metric="haversine"
        )

        elevated["hotspot_cluster"] = (
            clustering.fit_predict(
                coordinates
            )
        )

        snapshot["hotspot_cluster"] = -1

        snapshot.loc[
            elevated.index,
            "hotspot_cluster"
        ] = elevated[
            "hotspot_cluster"
        ]

    else:

        snapshot["hotspot_cluster"] = -1

    # --------------------------------------------------
    # Sort and save
    # --------------------------------------------------

    snapshot = snapshot.sort_values(
        "hotspot_score",
        ascending=False
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    snapshot.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n==============================")
    print("HOTSPOT DETECTION RESULTS")
    print("==============================")

    print(
        "High-risk locations:",
        (
            snapshot["risk_level"] == "HIGH"
        ).sum()
    )

    print(
        "Moderate-risk locations:",
        (
            snapshot["risk_level"] == "MODERATE"
        ).sum()
    )

    print(
        "Low-risk locations:",
        (
            snapshot["risk_level"] == "LOW"
        ).sum()
    )

    print(
        "\nTop 15 hotspot candidates:"
    )

    print(
        snapshot[
            [
                "station_id",
                "station",
                "latitude",
                "longitude",
                "pm25",
                "pm10",
                "no2",
                "wind_speed_10m",
                "pm25_anomaly",
                "hotspot_score",
                "risk_level",
                "hotspot_cluster"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()