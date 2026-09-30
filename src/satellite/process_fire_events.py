from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRMS_FILE = (
    PROJECT_ROOT
    / "data/raw/firms_india_viirs_overlap.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "firms_fire_events_processed.csv"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data/reference/ne_10m_admin_0_countries.geojson"
)


# ============================================================
# FIRE EVENT SETTINGS
# ============================================================

# Satellite detections within roughly 5 km can belong
# to the same spatial fire footprint.
SPATIAL_RADIUS_KM = 5.0

# A single physical event is limited to 3 hours.
MAX_EVENT_DURATION_HOURS = 3.0


# ============================================================
# LOAD FIRMS DATA
# ============================================================

def load_fire_data():

    if not FIRMS_FILE.exists():
        raise FileNotFoundError(
            f"FIRMS overlap file not found: {FIRMS_FILE}"
        )

    df = pd.read_csv(
        FIRMS_FILE
    )

    required_columns = [
        "latitude",
        "longitude",
        "acquisition_datetime",
        "satellite",
        "instrument",
        "confidence",
        "frp",
        "daynight",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing FIRMS columns: {missing}"
        )

    # --------------------------------------------------------
    # TYPE CONVERSION
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df["acquisition_datetime"] = pd.to_datetime(
        df["acquisition_datetime"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # REMOVE INVALID RECORDS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acquisition_datetime",
        ]
    ).copy()

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = (
        df.sort_values(
            "acquisition_datetime"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CLASSIFY INDIA / OUTSIDE INDIA
# ============================================================

def classify_fire_locations(df):

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"10m country boundary file not found: "
            f"{BOUNDARY_FILE}"
        )

    print(
        "Loading Natural Earth 10m country boundaries..."
    )

    countries = gpd.read_file(
        BOUNDARY_FILE
    )

    if countries.empty:
        raise RuntimeError(
            "Country boundary dataset is empty."
        )

    # --------------------------------------------------------
    # SELECT INDIA ONLY
    # --------------------------------------------------------

    if "ISO_A3" in countries.columns:

        india = countries[
            countries["ISO_A3"] == "IND"
        ].copy()

    else:

        india = countries[
            countries["ADMIN"] == "India"
        ].copy()

    # Fallback if ISO_A3 does not contain IND
    if india.empty and "ADMIN" in countries.columns:

        india = countries[
            countries["ADMIN"] == "India"
        ].copy()

    if india.empty:
        raise RuntimeError(
            "India could not be found in the "
            "10m country boundary dataset."
        )

    india = india.to_crs(
        "EPSG:4326"
    )

    # --------------------------------------------------------
    # MERGE INDIA GEOMETRIES
    # --------------------------------------------------------

    india_geometry = india.geometry.union_all()

    # --------------------------------------------------------
    # CREATE POINT GEOMETRIES
    # --------------------------------------------------------

    fire_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )

    # --------------------------------------------------------
    # COUNTRY MEMBERSHIP
    # --------------------------------------------------------

    inside_india = (
        fire_gdf.geometry.intersects(
            india_geometry
        )
    )

    fire_gdf["location_scope"] = np.where(
        inside_india,
        "INDIA",
        "OUTSIDE_INDIA",
    )

    return pd.DataFrame(
        fire_gdf.drop(
            columns=["geometry"],
            errors="ignore",
        )
    )


# ============================================================
# SPATIAL CLUSTERING WITHIN EACH DAY
# ============================================================

def spatial_cluster_one_day(
    day_df,
    date_label,
):
    """
    Cluster VIIRS detections spatially within one calendar day.

    DBSCAN uses haversine distance, so the radius is geographic.
    min_samples=1 means even a single valid satellite detection
    can represent a fire event.
    """

    if day_df.empty:
        return day_df.copy()

    data = (
        day_df
        .sort_values(
            "acquisition_datetime"
        )
        .reset_index(drop=True)
        .copy()
    )

    coordinates_radians = np.radians(
        data[
            [
                "latitude",
                "longitude",
            ]
        ].to_numpy()
    )

    earth_radius_km = 6371.0088

    eps_radians = (
        SPATIAL_RADIUS_KM
        / earth_radius_km
    )

    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=1,
        metric="haversine",
        algorithm="ball_tree",
    )

    labels = dbscan.fit_predict(
        coordinates_radians
    )

    data["spatial_cluster"] = labels

    data["cluster_date"] = str(
        date_label
    )

    return data


# ============================================================
# SPLIT SPATIAL CLUSTERS BY TIME
# ============================================================

def split_cluster_by_time(
    cluster_df,
):
    """
    A spatial cluster may contain repeated satellite
    observations throughout the day.

    Split it into separate fire events whenever:
        1. the next detection is more than 3 hours after
           the current event start, OR
        2. the event would exceed the maximum duration.

    This prevents one event from incorrectly spanning
    several days because of transitive spatial chaining.
    """

    data = (
        cluster_df
        .sort_values(
            "acquisition_datetime"
        )
        .reset_index(drop=True)
        .copy()
    )

    event_number = 0
    event_labels = []

    current_start = None

    for timestamp in data[
        "acquisition_datetime"
    ]:

        if current_start is None:

            event_number += 1
            current_start = timestamp

        else:

            elapsed_hours = (
                timestamp
                - current_start
            ).total_seconds() / 3600.0

            if (
                elapsed_hours
                > MAX_EVENT_DURATION_HOURS
            ):

                event_number += 1
                current_start = timestamp

        event_labels.append(
            event_number
        )

    data["temporal_event"] = (
        event_labels
    )

    return data


# ============================================================
# BUILD FIRE EVENTS
# ============================================================

def cluster_fire_detections(
    df,
):
    """
    Create physically and temporally bounded satellite fire
    events from individual VIIRS detections.
    """

    if df.empty:
        return pd.DataFrame()

    daily_clusters = []

    # --------------------------------------------------------
    # SPATIAL CLUSTERING PER CALENDAR DAY
    # --------------------------------------------------------

    df = df.copy()

    df["calendar_date"] = (
        df["acquisition_datetime"]
        .dt.date
    )

    print(
        "\nSpatially clustering daily satellite detections..."
    )

    for date_value, day_df in df.groupby(
        "calendar_date"
    ):

        clustered_day = spatial_cluster_one_day(
            day_df,
            date_value,
        )

        daily_clusters.append(
            clustered_day
        )

    spatial_data = pd.concat(
        daily_clusters,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    final_groups = []

    for (
        calendar_date,
        spatial_cluster,
    ), group in spatial_data.groupby(
        [
            "calendar_date",
            "spatial_cluster",
        ]
    ):

        split_group = split_cluster_by_time(
            group
        )

        final_groups.append(
            split_group
        )

    clustered = pd.concat(
        final_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # BUILD UNIQUE INTERNAL EVENT KEY
    # --------------------------------------------------------

    clustered["event_group_key"] = (
        clustered[
            "calendar_date"
        ].astype(str)
        + "_"
        + clustered[
            "spatial_cluster"
        ].astype(str)
        + "_"
        + clustered[
            "temporal_event"
        ].astype(str)
    )

    return clustered


# ============================================================
# FIRMS CONFIDENCE
# ============================================================

def confidence_score(
    confidence,
):

    mapping = {
        "h": 1.0,
        "n": 0.6,
        "l": 0.3,
    }

    return mapping.get(
        str(confidence).lower(),
        0.5,
    )


# ============================================================
# BUILD AGGREGATED FIRE EVENTS
# ============================================================

def build_fire_events(
    clustered,
):

    if clustered.empty:
        return pd.DataFrame()

    clustered = clustered.copy()

    clustered["confidence_numeric"] = (
        clustered[
            "confidence"
        ]
        .apply(
            confidence_score
        )
    )

    event_rows = []

    grouped = clustered.groupby(
        "event_group_key"
    )

    for internal_key, group in grouped:

        group = (
            group
            .sort_values(
                "acquisition_datetime"
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        start_time = (
            group[
                "acquisition_datetime"
            ].min()
        )

        end_time = (
            group[
                "acquisition_datetime"
            ].max()
        )

        duration_hours = (
            end_time - start_time
        ).total_seconds() / 3600.0

        # ----------------------------------------------------
        # CENTROID
        # ----------------------------------------------------

        event_latitude = (
            group[
                "latitude"
            ].mean()
        )

        event_longitude = (
            group[
                "longitude"
            ].mean()
        )

        # ----------------------------------------------------
        # SCOPE
        # ----------------------------------------------------

        scopes = set(
            group[
                "location_scope"
            ]
        )

        if scopes == {"INDIA"}:

            event_scope = "INDIA"

        elif scopes == {
            "OUTSIDE_INDIA"
        }:

            event_scope = "OUTSIDE_INDIA"

        else:

            event_scope = "CROSS_BORDER"

        # ----------------------------------------------------
        # SATELLITES
        # ----------------------------------------------------

        satellites = sorted(
            group[
                "satellite"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # ----------------------------------------------------
        # FRP
        # ----------------------------------------------------

        frp_values = pd.to_numeric(
            group["frp"],
            errors="coerce",
        )

        max_frp = (
            frp_values.max()
        )

        mean_frp = (
            frp_values.mean()
        )

        total_frp = (
            frp_values
            .fillna(0)
            .sum()
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_values = (
            group[
                "confidence"
            ]
            .astype(str)
            .str.lower()
        )

        if (
            confidence_values
            .eq("h")
            .any()
        ):

            strongest_confidence = "h"

        elif (
            confidence_values
            .eq("n")
            .any()
        ):

            strongest_confidence = "n"

        else:

            strongest_confidence = "l"

        max_confidence_score = (
            group[
                "confidence_numeric"
            ].max()
        )

        # ----------------------------------------------------
        # DAY / NIGHT
        # ----------------------------------------------------

        daynight = ",".join(
            sorted(
                group[
                    "daynight"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        event_rows.append(
            {
                "internal_event_key": internal_key,
                "event_latitude": event_latitude,
                "event_longitude": event_longitude,
                "event_start": start_time,
                "event_end": end_time,
                "event_timestamp": start_time,
                "duration_hours": duration_hours,
                "detection_count": len(group),
                "satellite_count": len(satellites),
                "satellites": ",".join(satellites),
                "max_frp": max_frp,
                "mean_frp": mean_frp,
                "total_frp": total_frp,
                "strongest_confidence": strongest_confidence,
                "confidence_score": max_confidence_score,
                "daynight": daynight,
                "fire_scope": event_scope,
            }
        )

    events = pd.DataFrame(
        event_rows
    )

    # --------------------------------------------------------
    # SATELLITE STRENGTH SCORE
    # --------------------------------------------------------

    if not events.empty:

        frp_rank = (
            events[
                "max_frp"
            ]
            .fillna(0)
            .rank(
                pct=True,
                method="average",
            )
        )

        events[
            "satellite_strength_score"
        ] = (
            0.70 * frp_rank
            + 0.30 * events[
                "confidence_score"
            ]
        )

        events[
            "satellite_strength_score"
        ] = (
            events[
                "satellite_strength_score"
            ]
            .clip(
                0,
                1,
            )
        )

    return events


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_fire_events():

    print("\n==============================")
    print("FIRMS FIRE EVENT PROCESSING")
    print("==============================")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    fires = load_fire_data()

    print(
        "Raw FIRMS detections:",
        len(fires),
    )

    print(
        "Raw satellite date range:"
    )

    print(
        fires[
            "acquisition_datetime"
        ].min(),
        "→",
        fires[
            "acquisition_datetime"
        ].max(),
    )

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    print(
        "\nClassifying fire detections..."
    )

    fires = classify_fire_locations(
        fires
    )

    print(
        "\nDetection location classification:"
    )

    print(
        fires[
            "location_scope"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # CLUSTER
    # --------------------------------------------------------

    clustered = cluster_fire_detections(
        fires
    )

    # --------------------------------------------------------
    # BUILD EVENTS
    # --------------------------------------------------------

    events = build_fire_events(
        clustered
    )

    if events.empty:

        raise RuntimeError(
            "No satellite fire events were generated."
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    events = (
        events
        .sort_values(
            [
                "event_timestamp",
                "max_frp",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # FINAL EVENT IDS
    # --------------------------------------------------------

    events["fire_event_id"] = [
        f"SATF-{index + 1:04d}"
        for index in range(
            len(events)
        )
    ]

    # Put ID first
    first_column = events.pop(
        "fire_event_id"
    )

    events.insert(
        0,
        "fire_event_id",
        first_column,
    )

    # --------------------------------------------------------
    # ROUND
    # --------------------------------------------------------

    round_columns = [
        "event_latitude",
        "event_longitude",
        "duration_hours",
        "max_frp",
        "mean_frp",
        "total_frp",
        "confidence_score",
        "satellite_strength_score",
    ]

    for column in round_columns:

        if column in events.columns:

            events[column] = (
                events[column]
                .round(3)
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("SATELLITE FIRE EVENT RESULTS")
    print("==============================")

    print(
        "Raw satellite detections:",
        len(fires),
    )

    print(
        "Grouped fire events:",
        len(events),
    )

    reduction = (
        1
        - (
            len(events)
            / len(fires)
        )
    ) * 100

    print(
        f"Detection-to-event reduction: "
        f"{reduction:.1f}%"
    )

    print(
        "\nFire event scope:"
    )

    print(
        events[
            "fire_scope"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSatellite coverage:"
    )

    print(
        events[
            "satellites"
        ]
        .value_counts()
        .head(10)
        .to_string()
    )

    print(
        "\nFire events by date:"
    )

    print(
        events[
            "event_start"
        ]
        .dt.date
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nEvent duration statistics:"
    )

    print(
        events[
            "duration_hours"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nHighest-FRP fire events:"
    )

    display_columns = [
        "fire_event_id",
        "event_latitude",
        "event_longitude",
        "event_timestamp",
        "event_end",
        "duration_hours",
        "detection_count",
        "satellites",
        "max_frp",
        "strongest_confidence",
        "fire_scope",
        "satellite_strength_score",
    ]

    print(
        events[
            display_columns
        ]
        .sort_values(
            "max_frp",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nSatellite fire-event processing completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    process_fire_events()