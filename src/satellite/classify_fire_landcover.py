from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRE_EVENTS_FILE = (
    PROJECT_ROOT
    / "data/processed/firms_fire_events_processed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fire_landcover_context.csv"
)


# ============================================================
# ESA WORLDCOVER SETTINGS
# ============================================================

WORLDCOVER_BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "v200/2021/map/"
)

# WorldCover 2021 classes
LANDCOVER_CLASSES = {
    10: "Tree Cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/Sparse Vegetation",
    70: "Snow/Ice",
    80: "Permanent Water",
    90: "Herbaceous Wetland",
    95: "Mangroves",
    100: "Moss/Lichen",
}


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# Analyze a 1 km radius around each fire-event centroid.
SAMPLE_RADIUS_METERS = 1000

# Minimum cropland share used to flag an agricultural
# land-use context.
CROPLAND_CONTEXT_THRESHOLD = 0.30

# Minimum share for a clearly cropland-dominant context.
CROPLAND_DOMINANT_THRESHOLD = 0.50

# Retry remote COG access a few times if necessary.
MAX_RETRIES = 3

# Pause between retries.
RETRY_DELAY_SECONDS = 2


# ============================================================
# TILE NAME
# ============================================================

def get_tile_name(
    latitude,
    longitude,
):
    """
    WorldCover uses 3° x 3° tile naming.

    Example:
        latitude  = 20.97
        longitude = 85.17

    -> N18E084
    """

    lat_tile = math.floor(
        float(latitude) / 3
    ) * 3

    lon_tile = math.floor(
        float(longitude) / 3
    ) * 3

    if lat_tile >= 0:

        lat_part = (
            f"N{lat_tile:02d}"
        )

    else:

        lat_part = (
            f"S{abs(lat_tile):02d}"
        )

    if lon_tile >= 0:

        lon_part = (
            f"E{lon_tile:03d}"
        )

    else:

        lon_part = (
            f"W{abs(lon_tile):03d}"
        )

    return (
        f"{lat_part}{lon_part}"
    )


# ============================================================
# TILE URL
# ============================================================

def get_tile_url(
    tile_name,
):
    return (
        WORLDCOVER_BASE_URL
        + "ESA_WorldCover_10m_2021_v200_"
        + tile_name
        + "_Map.tif"
    )


# ============================================================
# LOAD FIRE EVENTS
# ============================================================

def load_fire_events():

    if not FIRE_EVENTS_FILE.exists():

        raise FileNotFoundError(
            f"Fire-events file not found: "
            f"{FIRE_EVENTS_FILE}"
        )

    events = pd.read_csv(
        FIRE_EVENTS_FILE
    )

    required_columns = [
        "fire_event_id",
        "event_latitude",
        "event_longitude",
        "event_timestamp",
        "fire_scope",
        "satellite_strength_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required event columns: {missing}"
        )

    events["event_latitude"] = pd.to_numeric(
        events["event_latitude"],
        errors="coerce",
    )

    events["event_longitude"] = pd.to_numeric(
        events["event_longitude"],
        errors="coerce",
    )

    events["satellite_strength_score"] = pd.to_numeric(
        events["satellite_strength_score"],
        errors="coerce",
    )

    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    events = events.dropna(
        subset=[
            "fire_event_id",
            "event_latitude",
            "event_longitude",
        ]
    ).copy()

    # Only Indian fire events need domestic
    # agricultural-land classification.
    events = events[
        events["fire_scope"] == "INDIA"
    ].copy()

    events = (
        events
        .sort_values(
            "event_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return events


# ============================================================
# OPEN REMOTE WORLDCOVER TILE
# ============================================================

def open_tile(
    tile_name,
):
    """
    Open a WorldCover COG through GDAL / VSICURL.

    The returned dataset should be closed by the caller.
    """

    tile_url = get_tile_url(
        tile_name
    )

    vsicurl_path = (
        "/vsicurl/"
        + tile_url
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            dataset = rasterio.open(
                vsicurl_path
            )

            return dataset

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                print(
                    f"Tile open failed "
                    f"(attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"Could not open WorldCover tile "
        f"{tile_name}: {last_error}"
    )


# ============================================================
# READ LAND-COVER WINDOW
# ============================================================

def read_event_landcover(
    dataset,
    latitude,
    longitude,
):
    """
    Read approximately a 1 km radius around a fire event
    and calculate land-cover class fractions.

    Returns:
        statistics dictionary
    """

    # --------------------------------------------------------
    # EVENT PIXEL
    # --------------------------------------------------------

    row, col = dataset.index(
        float(longitude),
        float(latitude),
    )

    # --------------------------------------------------------
    # PIXEL SIZE
    # --------------------------------------------------------

    pixel_width_degrees = abs(
        dataset.transform.a
    )

    pixel_height_degrees = abs(
        dataset.transform.e
    )

    # Approximate degree-to-metre conversion.
    # Longitude spacing changes with latitude.
    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0
        * max(
            0.05,
            math.cos(
                math.radians(
                    float(latitude)
                )
            ),
        )
    )

    meters_per_pixel_lat = (
        pixel_height_degrees
        * meters_per_degree_lat
    )

    meters_per_pixel_lon = (
        pixel_width_degrees
        * meters_per_degree_lon
    )

    radius_rows = max(
        1,
        int(
            SAMPLE_RADIUS_METERS
            / meters_per_pixel_lat
        ),
    )

    radius_cols = max(
        1,
        int(
            SAMPLE_RADIUS_METERS
            / meters_per_pixel_lon
        ),
    )

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    row_start = max(
        0,
        row - radius_rows,
    )

    row_stop = min(
        dataset.height,
        row + radius_rows + 1,
    )

    col_start = max(
        0,
        col - radius_cols,
    )

    col_stop = min(
        dataset.width,
        col + radius_cols + 1,
    )

    window = (
        (
            row_start,
            row_stop,
        ),
        (
            col_start,
            col_stop,
        ),
    )

    values = dataset.read(
        1,
        window=window,
    )

    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid = values[
        np.isin(
            values,
            list(
                LANDCOVER_CLASSES.keys()
            ),
        )
    ]

    expected_pixels = (
        (
            row + radius_rows + 1
            - (row - radius_rows)
        )
        *
        (
            col + radius_cols + 1
            - (col - radius_cols)
        )
    )

    coverage_fraction = (
        len(valid)
        / max(
            1,
            expected_pixels,
        )
    )

    if len(valid) == 0:

        raise ValueError(
            "No valid WorldCover pixels "
            "were found around event."
        )

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    unique_values, counts = np.unique(
        valid,
        return_counts=True,
    )

    total_valid = counts.sum()

    fractions = {}

    for value, count in zip(
        unique_values,
        counts,
    ):

        class_name = LANDCOVER_CLASSES.get(
            int(value),
            "Unknown",
        )

        fraction = (
            float(count)
            / float(total_valid)
        )

        fractions[int(value)] = {
            "name": class_name,
            "fraction": fraction,
        }

    # --------------------------------------------------------
    # FRACTIONS BY CLASS
    # --------------------------------------------------------

    def fraction_for(
        class_code,
    ):

        if class_code not in fractions:
            return 0.0

        return fractions[
            class_code
        ]["fraction"]

    cropland_fraction = fraction_for(
        40
    )

    forest_fraction = fraction_for(
        10
    )

    shrubland_fraction = fraction_for(
        20
    )

    grassland_fraction = fraction_for(
        30
    )

    builtup_fraction = fraction_for(
        50
    )

    bare_fraction = fraction_for(
        60
    )

    water_fraction = fraction_for(
        80
    )

    wetland_fraction = fraction_for(
        90
    )

    mangrove_fraction = fraction_for(
        95
    )

    moss_fraction = fraction_for(
        100
    )

    # --------------------------------------------------------
    # DOMINANT CLASS
    # --------------------------------------------------------

    dominant_code = max(
        fractions,
        key=lambda code:
        fractions[code]["fraction"],
    )

    dominant_name = (
        LANDCOVER_CLASSES.get(
            dominant_code,
            "Unknown",
        )
    )

    dominant_fraction = (
        fractions[
            dominant_code
        ]["fraction"]
    )

    # --------------------------------------------------------
    # LAND-COVER CONTEXT
    # --------------------------------------------------------

    if (
        cropland_fraction
        >= CROPLAND_DOMINANT_THRESHOLD
    ):

        landcover_context = (
            "CROPLAND_DOMINANT"
        )

    elif (
        cropland_fraction
        >= CROPLAND_CONTEXT_THRESHOLD
    ):

        landcover_context = (
            "CROPLAND_PRESENT"
        )

    elif (
        builtup_fraction
        >= 0.50
    ):

        landcover_context = (
            "BUILT_UP_DOMINANT"
        )

    elif (
        forest_fraction
        + shrubland_fraction
        + grassland_fraction
        >= 0.50
    ):

        landcover_context = (
            "NATURAL_VEGETATION_DOMINANT"
        )

    else:

        landcover_context = (
            "MIXED_OR_OTHER"
        )

    # --------------------------------------------------------
    # AGRICULTURAL CONTEXT
    # --------------------------------------------------------

    agricultural_context = (
        cropland_fraction
        >= CROPLAND_CONTEXT_THRESHOLD
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "worldcover_tile": None,
        "dominant_landcover_code": dominant_code,
        "dominant_landcover": dominant_name,
        "dominant_landcover_fraction": dominant_fraction,
        "cropland_fraction": cropland_fraction,
        "forest_fraction": forest_fraction,
        "shrubland_fraction": shrubland_fraction,
        "grassland_fraction": grassland_fraction,
        "builtup_fraction": builtup_fraction,
        "bare_sparse_fraction": bare_fraction,
        "water_fraction": water_fraction,
        "wetland_fraction": wetland_fraction,
        "mangrove_fraction": mangrove_fraction,
        "moss_lichen_fraction": moss_fraction,
        "valid_pixel_count": int(total_valid),
        "sample_coverage_fraction": coverage_fraction,
        "landcover_context": landcover_context,
        "agricultural_context": agricultural_context,
    }


# ============================================================
# PROCESS EVENTS
# ============================================================

def process_landcover():

    print("\n==============================")
    print("FIRE EVENT LAND-COVER ANALYSIS")
    print("==============================")

    events = load_fire_events()

    print(
        "Indian fire events to classify:",
        len(events),
    )

    # --------------------------------------------------------
    # TILE ASSIGNMENT
    # --------------------------------------------------------

    events["worldcover_tile"] = events.apply(
        lambda row: get_tile_name(
            row["event_latitude"],
            row["event_longitude"],
        ),
        axis=1,
    )

    print(
        "WorldCover tiles required:",
        events[
            "worldcover_tile"
        ].nunique(),
    )

    # --------------------------------------------------------
    # PROCESS TILE BY TILE
    # --------------------------------------------------------

    all_results = []

    open_datasets = {}

    try:

        for tile_name, tile_events in events.groupby(
            "worldcover_tile"
        ):

            print("\n------------------------------")
            print(
                f"Processing tile {tile_name}"
            )
            print(
                f"Events in tile: {len(tile_events)}"
            )
            print("------------------------------")

            # ------------------------------------------------
            # OPEN COG
            # ------------------------------------------------

            try:

                dataset = open_tile(
                    tile_name
                )

                open_datasets[
                    tile_name
                ] = dataset

            except Exception as exc:

                print(
                    f"ERROR opening tile "
                    f"{tile_name}: {exc}"
                )

                # Record failed events instead of stopping
                for _, event in tile_events.iterrows():

                    all_results.append(
                        {
                            "fire_event_id": event[
                                "fire_event_id"
                            ],
                            "worldcover_tile": tile_name,
                            "classification_status": "FAILED",
                            "classification_error": str(
                                exc
                            ),
                        }
                    )

                continue

            # ------------------------------------------------
            # PROCESS EVENTS
            # ------------------------------------------------

            for _, event in tile_events.iterrows():

                try:

                    statistics = (
                        read_event_landcover(
                            dataset,
                            event[
                                "event_latitude"
                            ],
                            event[
                                "event_longitude"
                            ],
                        )
                    )

                    statistics[
                        "fire_event_id"
                    ] = event[
                        "fire_event_id"
                    ]

                    statistics[
                        "worldcover_tile"
                    ] = tile_name

                    statistics[
                        "classification_status"
                    ] = "SUCCESS"

                    statistics[
                        "classification_error"
                    ] = ""

                    all_results.append(
                        statistics
                    )

                except Exception as exc:

                    print(
                        f"ERROR processing "
                        f"{event['fire_event_id']}: "
                        f"{exc}"
                    )

                    all_results.append(
                        {
                            "fire_event_id": event[
                                "fire_event_id"
                            ],
                            "worldcover_tile": tile_name,
                            "classification_status": "FAILED",
                            "classification_error": str(
                                exc
                            ),
                        }
                    )

            print(
                f"Completed tile {tile_name}"
            )

    finally:

        # ----------------------------------------------------
        # CLOSE ALL REMOTE DATASETS
        # ----------------------------------------------------

        for dataset in (
            open_datasets.values()
        ):

            try:

                dataset.close()

            except Exception:
                pass

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    if results.empty:

        raise RuntimeError(
            "No land-cover classification "
            "results were produced."
        )

    # --------------------------------------------------------
    # MERGE EVENT METADATA
    # --------------------------------------------------------

    event_metadata = events[
        [
            "fire_event_id",
            "event_latitude",
            "event_longitude",
            "event_timestamp",
            "detection_count"
            if "detection_count" in events.columns
            else "fire_event_id",
            "max_frp"
            if "max_frp" in events.columns
            else "fire_event_id",
            "satellite_strength_score",
        ]
    ].copy()

    # Remove duplicate placeholder columns
    event_metadata = (
        event_metadata.loc[
            :,
            ~event_metadata.columns.duplicated()
        ]
    )

    results = event_metadata.merge(
        results,
        on="fire_event_id",
        how="left",
    )

    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    results = (
        results
        .sort_values(
            "event_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("LAND-COVER RESULTS")
    print("==============================")

    success = (
        results[
            "classification_status"
        ]
        == "SUCCESS"
    )

    print(
        "Total Indian fire events:",
        len(results),
    )

    print(
        "Successfully classified:",
        int(success.sum()),
    )

    print(
        "Failed classifications:",
        int((~success).sum()),
    )

    successful = results[
        success
    ].copy()

    if not successful.empty:

        print(
            "\nLand-cover context:"
        )

        print(
            successful[
                "landcover_context"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nAgricultural context:"
        )

        print(
            successful[
                "agricultural_context"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nCropland fraction statistics:"
        )

        print(
            successful[
                "cropland_fraction"
            ]
            .describe()
            .to_string()
        )

        print(
            "\nHighest cropland-context events:"
        )

        display_columns = [
            "fire_event_id",
            "event_latitude",
            "event_longitude",
            "event_timestamp",
            "dominant_landcover",
            "dominant_landcover_fraction",
            "cropland_fraction",
            "forest_fraction",
            "grassland_fraction",
            "builtup_fraction",
            "landcover_context",
            "agricultural_context",
        ]

        print(
            successful[
                display_columns
            ]
            .sort_values(
                "cropland_fraction",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nFire-event land-cover analysis completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    process_landcover()