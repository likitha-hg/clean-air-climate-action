from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans


INPUT_FILE = Path(
    "data/processed/india_station_inventory.csv"
)

OUTPUT_FILE = Path(
    "data/processed/training_station_candidates.csv"
)

N_STATIONS = 50


def select_training_stations():

    df = pd.read_csv(INPUT_FILE)

    # Convert station-level last-data timestamp
    df["datetime_last"] = pd.to_datetime(
        df["datetime_last"],
        utc=True,
        errors="coerce"
    )

    # Keep only stations with all six pollutants
    # and station-level data extending into 2026.
    candidates = df[
        (df["all_six_pollutants"] == True)
        & (df["datetime_last"] >= "2026-01-01")
    ].copy()

    candidates = candidates.dropna(
        subset=["latitude", "longitude"]
    )

    print(
        "Current six-pollutant stations available:",
        len(candidates)
    )

    if len(candidates) < N_STATIONS:
        raise ValueError(
            f"Only {len(candidates)} stations available. "
            f"Cannot select {N_STATIONS}."
        )

    # Geographic clustering so we don't accidentally
    # select 50 stations concentrated in a few cities.
    coordinates = candidates[
        ["latitude", "longitude"]
    ]

    kmeans = KMeans(
        n_clusters=N_STATIONS,
        random_state=42,
        n_init=10
    )

    candidates["cluster"] = kmeans.fit_predict(
        coordinates
    )

    selected_rows = []

    for cluster_id in range(N_STATIONS):

        cluster = candidates[
            candidates["cluster"] == cluster_id
        ].copy()

        if cluster.empty:
            continue

        center = kmeans.cluster_centers_[cluster_id]

        # Select the station closest to cluster center.
        cluster["distance_to_center"] = (
            (cluster["latitude"] - center[0]) ** 2
            + (cluster["longitude"] - center[1]) ** 2
        ) ** 0.5

        selected = cluster.sort_values(
            ["distance_to_center", "datetime_last"],
            ascending=[True, False]
        ).iloc[0]

        selected_rows.append(selected)

    selected_df = pd.DataFrame(
        selected_rows
    ).drop_duplicates(
        subset=["station_id"]
    )

    # If clustering produced fewer than 50 unique stations,
    # fill the remaining slots with the most recent candidates.
    if len(selected_df) < N_STATIONS:

        remaining = candidates[
            ~candidates["station_id"].isin(
                selected_df["station_id"]
            )
        ].sort_values(
            "datetime_last",
            ascending=False
        )

        needed = N_STATIONS - len(selected_df)

        selected_df = pd.concat(
            [
                selected_df,
                remaining.head(needed)
            ],
            ignore_index=True
        )

    selected_df = selected_df.sort_values(
        ["latitude", "longitude"]
    ).reset_index(drop=True)

    selected_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSelected training stations:",
        len(selected_df)
    )

    print(
        "\nTraining station list:"
    )

    print(
        selected_df[
            [
                "station_id",
                "station",
                "locality",
                "latitude",
                "longitude",
                "datetime_first",
                "datetime_last"
            ]
        ].to_string(index=False)
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    select_training_stations()