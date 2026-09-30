from pathlib import Path

import pandas as pd

from src.satellite.firms_client import (
    get_firms_fire_data,
    INDIA_BBOX,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data/raw"

OUTPUT_FILE = (
    OUTPUT_DIR / "firms_india_viirs_overlap.csv"
)


# ============================================================
# SYNCHRONIZED ANALYSIS WINDOW
# ============================================================

START_DATE = "2026-09-13"
DAY_RANGE = 5

SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_overlap_data():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_data = []

    print("\n==============================")
    print("FIRMS OVERLAP DATA DOWNLOAD")
    print("==============================")

    print(
        f"Analysis window: {START_DATE} "
        f"to 2026-09-17"
    )

    for source in SOURCES:

        print("\n------------------------------")
        print(f"Source: {source}")
        print("------------------------------")

        df = get_firms_fire_data(
            source=source,
            bbox=INDIA_BBOX,
            day_range=DAY_RANGE,
            date=START_DATE,
        )

        if df.empty:

            print(
                f"No detections returned for {source}."
            )

            continue

        df["firms_source"] = source

        all_data.append(df)

        print(
            f"Detections returned: {len(df)}"
        )

    if not all_data:

        raise RuntimeError(
            "No FIRMS detections were returned."
        )

    # ========================================================
    # COMBINE SATELLITES
    # ========================================================

    fire_data = pd.concat(
        all_data,
        ignore_index=True,
    )

    # ========================================================
    # BUILD TIMESTAMP
    # ========================================================

    fire_data["acq_time"] = (
        fire_data["acq_time"]
        .astype(str)
        .str.zfill(4)
    )

    fire_data["acquisition_datetime"] = pd.to_datetime(
        fire_data["acq_date"].astype(str)
        + " "
        + fire_data["acq_time"].str[:2]
        + ":"
        + fire_data["acq_time"].str[2:],
        errors="coerce",
        utc=True,
    )

    # ========================================================
    # NUMERIC FIELDS
    # ========================================================

    numeric_columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "frp",
    ]

    for column in numeric_columns:

        if column in fire_data.columns:

            fire_data[column] = pd.to_numeric(
                fire_data[column],
                errors="coerce",
            )

    # ========================================================
    # REMOVE INVALID RECORDS
    # ========================================================

    fire_data = fire_data.dropna(
        subset=[
            "latitude",
            "longitude",
            "acquisition_datetime",
        ]
    ).copy()

    # ========================================================
    # REMOVE EXACT DUPLICATES
    # ========================================================

    before = len(fire_data)

    fire_data = fire_data.drop_duplicates(
        subset=[
            "latitude",
            "longitude",
            "acquisition_datetime",
            "satellite",
            "instrument",
        ]
    ).copy()

    duplicates_removed = (
        before - len(fire_data)
    )

    # ========================================================
    # SORT
    # ========================================================

    fire_data = (
        fire_data
        .sort_values(
            "acquisition_datetime"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    fire_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("FIRMS OVERLAP DATASET")
    print("==============================")

    print(
        "Total detections:",
        len(fire_data),
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    print(
        "Actual timestamp range:"
    )

    print(
        fire_data["acquisition_datetime"].min(),
        "→",
        fire_data["acquisition_datetime"].max(),
    )

    print(
        "\nSatellite distribution:"
    )

    print(
        fire_data["satellite"]
        .value_counts()
        .to_string()
    )

    print(
        "\nDetections by date:"
    )

    print(
        fire_data["acq_date"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nHighest FRP detections:"
    )

    display_columns = [
        "latitude",
        "longitude",
        "acquisition_datetime",
        "satellite",
        "confidence",
        "frp",
        "daynight",
    ]

    print(
        fire_data[
            display_columns
        ]
        .sort_values(
            "frp",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nFIRMS overlap download completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    download_overlap_data()