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

OUTPUT_FILE = OUTPUT_DIR / "firms_india_viirs_latest.csv"


# ============================================================
# SETTINGS
# ============================================================

SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

DAY_RANGE = 5


# ============================================================
# DOWNLOAD
# ============================================================

def download_all_sources():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_data = []

    for source in SOURCES:

        print("\n==============================")
        print(f"DOWNLOADING {source}")
        print("==============================")

        df = get_firms_fire_data(
            source=source,
            bbox=INDIA_BBOX,
            day_range=DAY_RANGE,
        )

        if df.empty:

            print(
                f"No detections returned for {source}."
            )

            continue

        df["firms_source"] = source

        all_data.append(df)

        print(
            f"{source} detections: {len(df)}"
        )

    if not all_data:

        raise RuntimeError(
            "No FIRMS data was returned from either satellite source."
        )

    # --------------------------------------------------------
    # COMBINE SOURCES
    # --------------------------------------------------------

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # BUILD ACQUISITION TIMESTAMP
    # --------------------------------------------------------

    combined["acq_time"] = (
        combined["acq_time"]
        .astype(str)
        .str.zfill(4)
    )

    combined["acquisition_datetime"] = pd.to_datetime(
        combined["acq_date"].astype(str)
        + " "
        + combined["acq_time"].str[:2]
        + ":"
        + combined["acq_time"].str[2:],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

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

        if column in combined.columns:

            combined[column] = pd.to_numeric(
                combined[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # REMOVE INVALID COORDINATES
    # --------------------------------------------------------

    combined = combined.dropna(
        subset=[
            "latitude",
            "longitude",
            "acquisition_datetime",
        ]
    )

    combined = combined[
        combined["latitude"].between(-90, 90)
        & combined["longitude"].between(-180, 180)
    ]

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=[
            "latitude",
            "longitude",
            "acquisition_datetime",
            "satellite",
            "instrument",
        ]
    )

    duplicates_removed = before - len(combined)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    combined = combined.sort_values(
        "acquisition_datetime"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n==============================")
    print("FIRMS SATELLITE DATASET")
    print("==============================")

    print(
        "Total detections:",
        len(combined),
    )

    print(
        "Exact duplicates removed:",
        duplicates_removed,
    )

    print(
        "Date range:",
        combined["acquisition_datetime"].min(),
        "→",
        combined["acquisition_datetime"].max(),
    )

    print("\nSatellite distribution:")

    print(
        combined["satellite"]
        .value_counts()
        .to_string()
    )

    print("\nConfidence distribution:")

    print(
        combined["confidence"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nHighest FRP detections:")

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
        combined[
            display_columns
        ]
        .sort_values(
            "frp",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    download_all_sources()