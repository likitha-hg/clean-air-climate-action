"""
Build corridor-associated monitoring membership.

Method:
    station
        -> state
        -> corridor state scope

A station may belong to multiple corridor-associated networks.

Important:
This is a geographic-scope association for the prototype.
It does NOT mean the station is physically located on the
official corridor route or at an official industrial node.
"""

from pathlib import Path

import pandas as pd

from src.corridors.registry import CORRIDORS
from src.corridors.station_mapping import STATION_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data/processed/pm25_forecast_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/corridor_station_membership.csv"
)


def build_membership():

    df = pd.read_csv(
        FORECAST_FILE,
        usecols=[
            "station_id",
            "station",
            "latitude",
            "longitude",
        ],
    )

    stations = (
        df
        .drop_duplicates(
            subset=["station_id"]
        )
        .copy()
    )

    records = []

    for _, row in stations.iterrows():

        station = row["station"]

        state = STATION_STATE.get(
            station
        )

        if state is None:
            raise KeyError(
                f"Station is not mapped to a state: {station}"
            )

        matched_corridors = []

        for corridor_id, corridor in CORRIDORS.items():

            if state in corridor["states"]:

                matched_corridors.append(
                    corridor_id
                )

                records.append(
                    {
                        "station_id":
                            row["station_id"],

                        "station":
                            station,

                        "latitude":
                            row["latitude"],

                        "longitude":
                            row["longitude"],

                        "state":
                            state,

                        "corridor_id":
                            corridor_id,

                        "corridor_name":
                            corridor["name"],

                        "association_type":
                            "STATE_SCOPE",
                    }
                )

        # Keep stations outside all five corridor scopes visible
        # for completeness. They are not assigned to a corridor.
        if not matched_corridors:

            records.append(
                {
                    "station_id":
                        row["station_id"],

                    "station":
                        station,

                    "latitude":
                        row["latitude"],

                    "longitude":
                        row["longitude"],

                    "state":
                        state,

                    "corridor_id":
                        None,

                    "corridor_name":
                        None,

                    "association_type":
                        "OUTSIDE_REGISTERED_SCOPE",
                }
            )

    membership = pd.DataFrame(
        records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    membership.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return membership


if __name__ == "__main__":

    membership = build_membership()

    print("=" * 80)
    print("CORRIDOR STATION MEMBERSHIP")
    print("=" * 80)

    print(
        f"Output rows: {len(membership):,}"
    )

    print(
        f"Unique stations: "
        f"{membership['station_id'].nunique():,}"
    )

    assigned = membership[
        membership["corridor_id"].notna()
    ]

    print(
        f"Stations with corridor association: "
        f"{assigned['station_id'].nunique():,}"
    )

    outside = membership[
        membership["corridor_id"].isna()
    ]

    print(
        f"Stations outside registered scopes: "
        f"{outside['station_id'].nunique():,}"
    )

    print("\nCORRIDOR COVERAGE:")

    coverage = (
        assigned
        .groupby(
            [
                "corridor_id",
                "corridor_name",
            ]
        )["station_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        coverage.to_string()
    )

    print("\nSTATIONS WITHOUT CORRIDOR ASSOCIATION:")

    for station in (
        outside[
            "station"
        ]
        .drop_duplicates()
        .sort_values()
    ):

        state = (
            outside.loc[
                outside["station"] == station,
                "state",
            ]
            .iloc[0]
        )

        print(
            f" - {station} ({state})"
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)
