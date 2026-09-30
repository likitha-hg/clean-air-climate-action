from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORRELATION_FILE = (
    PROJECT_ROOT
    / "data/processed/fire_pollution_events.csv"
)

LANDCOVER_FILE = (
    PROJECT_ROOT
    / "data/processed/fire_landcover_context.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pollution_event_intelligence.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# WorldCover product used for this context layer.
LANDCOVER_YEAR = 2021


# ------------------------------------------------------------
# AGRICULTURAL BURNING RULES
# ------------------------------------------------------------

# Minimum cropland fraction for meaningful agricultural context.
MIN_CROPLAND_CONTEXT = 0.30

# Strong cropland context.
STRONG_CROPLAND_CONTEXT = 0.50

# Minimum PM2.5 elevation for agricultural event classification.
MIN_AGRICULTURAL_PM25_SCORE = 0.30

# Minimum combined event score.
MIN_AGRICULTURAL_EVENT_SCORE = 0.55


# ------------------------------------------------------------
# LOCAL BURNING RULE
# ------------------------------------------------------------

LOCAL_FIRE_RADIUS_KM = 50.0


# ============================================================
# LOAD CORRELATION DATA
# ============================================================

def load_correlation_data():

    if not CORRELATION_FILE.exists():

        raise FileNotFoundError(
            f"Correlation file not found: "
            f"{CORRELATION_FILE}"
        )

    df = pd.read_csv(
        CORRELATION_FILE
    )

    required_columns = [
        "correlation_id",
        "fire_event_id",
        "fire_scope",
        "fire_latitude",
        "fire_longitude",
        "fire_timestamp",
        "fire_max_frp",
        "fire_confidence",
        "station_id",
        "station",
        "distance_km",
        "time_lag_hours",
        "pm25_at_response",
        "pm25_zscore",
        "pm25_elevation_score",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_alignment_score",
        "kinematic_travel_time_hours",
        "transport_time_score",
        "transport_plausible",
        "event_score",
        "evidence_level",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Correlation file is missing columns: {missing}"
        )

    # --------------------------------------------------------
    # TYPES
    # --------------------------------------------------------

    df["fire_timestamp"] = pd.to_datetime(
        df["fire_timestamp"],
        errors="coerce",
        utc=True,
    )

    numeric_columns = [
        "fire_latitude",
        "fire_longitude",
        "fire_max_frp",
        "distance_km",
        "time_lag_hours",
        "pm25_at_response",
        "pm25_zscore",
        "pm25_elevation_score",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_alignment_score",
        "kinematic_travel_time_hours",
        "transport_time_score",
        "event_score",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# LOAD LAND-COVER DATA
# ============================================================

def load_landcover_data():

    if not LANDCOVER_FILE.exists():

        raise FileNotFoundError(
            f"Land-cover file not found: "
            f"{LANDCOVER_FILE}"
        )

    df = pd.read_csv(
        LANDCOVER_FILE
    )

    required_columns = [
        "fire_event_id",
        "worldcover_tile",
        "dominant_landcover",
        "dominant_landcover_fraction",
        "cropland_fraction",
        "forest_fraction",
        "shrubland_fraction",
        "grassland_fraction",
        "builtup_fraction",
        "bare_sparse_fraction",
        "water_fraction",
        "wetland_fraction",
        "mangrove_fraction",
        "moss_lichen_fraction",
        "valid_pixel_count",
        "sample_coverage_fraction",
        "landcover_context",
        "agricultural_context",
        "classification_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Land-cover file is missing columns: {missing}"
        )

    numeric_columns = [
        "dominant_landcover_fraction",
        "cropland_fraction",
        "forest_fraction",
        "shrubland_fraction",
        "grassland_fraction",
        "builtup_fraction",
        "bare_sparse_fraction",
        "water_fraction",
        "wetland_fraction",
        "mangrove_fraction",
        "moss_lichen_fraction",
        "valid_pixel_count",
        "sample_coverage_fraction",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # One land-cover record per fire event.
    df = (
        df.sort_values(
            "cropland_fraction",
            ascending=False,
        )
        .drop_duplicates(
            "fire_event_id"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# FIRE CONFIDENCE SCORE
# ============================================================

def fire_confidence_score(
    confidence,
):
    """
    Convert FIRMS categorical confidence into a normalized
    evidence value.

    h = high
    n = nominal
    l = low
    """

    mapping = {
        "h": 1.00,
        "n": 0.60,
        "l": 0.30,
    }

    return mapping.get(
        str(confidence).lower(),
        0.50,
    )


# ============================================================
# AGRICULTURAL BURNING SCORE
# ============================================================

def calculate_agricultural_score(
    cropland_fraction,
    pm25_elevation_score,
    event_score,
    fire_confidence,
):
    """
    Estimate strength of agricultural-burning context.

    This is an evidence score, NOT a source-attribution model.

    Inputs:
        cropland context
        PM2.5 anomaly
        multi-signal event score
        satellite detection confidence
    """

    cropland_fraction = float(
        np.clip(
            cropland_fraction
            if not pd.isna(cropland_fraction)
            else 0.0,
            0.0,
            1.0,
        )
    )

    pm25_elevation_score = float(
        np.clip(
            pm25_elevation_score
            if not pd.isna(pm25_elevation_score)
            else 0.0,
            0.0,
            1.0,
        )
    )

    event_score = float(
        np.clip(
            event_score
            if not pd.isna(event_score)
            else 0.0,
            0.0,
            1.0,
        )
    )

    satellite_confidence = fire_confidence_score(
        fire_confidence
    )

    score = (
        0.30 * cropland_fraction
        + 0.35 * pm25_elevation_score
        + 0.25 * event_score
        + 0.10 * satellite_confidence
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# SOURCE CONTEXT
# ============================================================

def classify_source_context(
    row,
):
    """
    Convert multiple evidence layers into a human-readable
    event context.

    The labels intentionally use 'potential' because the data
    do not prove physical source attribution.
    """

    fire_scope = str(
        row["fire_scope"]
    )

    distance_km = float(
        row["distance_km"]
    )

    event_score = float(
        row["event_score"]
    )

    pm25_elevation = float(
        row["pm25_elevation_score"]
    )

    cropland_fraction = float(
        row["cropland_fraction"]
        if not pd.isna(
            row["cropland_fraction"]
        )
        else 0.0
    )

    agricultural_score = float(
        row["agricultural_burning_score"]
    )

    transport_plausible = bool(
        row["transport_plausible"]
    )

    wind_alignment = float(
        row["wind_alignment_score"]
        if not pd.isna(
            row["wind_alignment_score"]
        )
        else 0.0
    )

    # --------------------------------------------------------
    # INDIA
    # --------------------------------------------------------

    if fire_scope == "INDIA":

        # Agricultural burning
        if (
            cropland_fraction
            >= STRONG_CROPLAND_CONTEXT
            and pm25_elevation
            >= MIN_AGRICULTURAL_PM25_SCORE
            and event_score
            >= MIN_AGRICULTURAL_EVENT_SCORE
        ):

            return (
                "POTENTIAL_AGRICULTURAL_BURNING"
            )

        # Moderate agricultural context
        if (
            cropland_fraction
            >= MIN_CROPLAND_CONTEXT
            and agricultural_score
            >= 0.50
            and pm25_elevation
            >= MIN_AGRICULTURAL_PM25_SCORE
        ):

            return (
                "POTENTIAL_AGRICULTURAL_BURNING"
            )

        # Very local fire without strong agricultural context
        if distance_km <= LOCAL_FIRE_RADIUS_KM:

            return (
                "POTENTIAL_LOCAL_BURNING"
            )

        # Domestic transport
        if (
            distance_km > LOCAL_FIRE_RADIUS_KM
            and transport_plausible
        ):

            return (
                "POTENTIAL_DOMESTIC_TRANSPORT"
            )

        return (
            "SATELLITE_FIRE_ASSOCIATION"
        )

    # --------------------------------------------------------
    # OUTSIDE INDIA
    # --------------------------------------------------------

    if fire_scope == "OUTSIDE_INDIA":

        if (
            transport_plausible
            and wind_alignment >= 0.60
        ):

            return (
                "POTENTIAL_REGIONAL_TRANSPORT"
            )

        return (
            "SATELLITE_FIRE_ASSOCIATION"
        )

    # --------------------------------------------------------
    # CROSS-BORDER
    # --------------------------------------------------------

    if fire_scope == "CROSS_BORDER":

        if (
            transport_plausible
            and wind_alignment >= 0.50
        ):

            return (
                "POTENTIAL_REGIONAL_TRANSPORT"
            )

        return (
            "SATELLITE_FIRE_ASSOCIATION"
        )

    return (
        "SATELLITE_FIRE_ASSOCIATION"
    )


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

def build_evidence_summary(
    row,
):
    """
    Generate a concise machine-generated evidence trail.
    """

    evidence = []

    # --------------------------------------------------------
    # SATELLITE
    # --------------------------------------------------------

    evidence.append(
        f"Satellite fire detected with "
        f"{row['fire_confidence']} confidence"
    )

    if not pd.isna(
        row["fire_max_frp"]
    ):

        evidence.append(
            f"maximum FRP {row['fire_max_frp']:.1f}"
        )

    # --------------------------------------------------------
    # LAND COVER
    # --------------------------------------------------------

    cropland = float(
        row["cropland_fraction"]
    )

    if cropland >= 0.50:

        evidence.append(
            f"surrounding land cover is "
            f"{cropland * 100:.1f}% cropland"
        )

    elif cropland >= 0.30:

        evidence.append(
            f"cropland is present in "
            f"{cropland * 100:.1f}% of the sampled area"
        )

    # --------------------------------------------------------
    # AIR QUALITY
    # --------------------------------------------------------

    pm25_elevation = float(
        row["pm25_elevation_score"]
    )

    if pm25_elevation >= 0.70:

        evidence.append(
            "strong PM2.5 elevation at a monitoring station"
        )

    elif pm25_elevation >= 0.40:

        evidence.append(
            "moderate PM2.5 elevation at a monitoring station"
        )

    elif pm25_elevation >= 0.20:

        evidence.append(
            "measurable PM2.5 elevation at a monitoring station"
        )

    # --------------------------------------------------------
    # SPATIAL
    # --------------------------------------------------------

    evidence.append(
        f"monitoring station {row['distance_km']:.1f} km "
        f"from fire event"
    )

    # --------------------------------------------------------
    # TEMPORAL
    # --------------------------------------------------------

    evidence.append(
        f"PM2.5 response observed "
        f"{row['time_lag_hours']:.1f} hours after fire detection"
    )

    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    if row["wind_alignment_score"] >= 0.70:

        evidence.append(
            "wind direction strongly supports transport"
        )

    elif row["wind_alignment_score"] >= 0.50:

        evidence.append(
            "wind direction supports transport"
        )

    # --------------------------------------------------------
    # TRANSPORT
    # --------------------------------------------------------

    if row["transport_plausible"]:

        if not pd.isna(
            row["kinematic_travel_time_hours"]
        ):

            evidence.append(
                "timing passes simple wind-based "
                "transport screening"
            )

    return "; ".join(
        evidence
    )


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(
    source_context,
):
    mapping = {

        "POTENTIAL_AGRICULTURAL_BURNING":
            (
                "Verify agricultural burning activity, "
                "monitor downwind stations, and assess "
                "whether additional satellite observations "
                "support the event."
            ),

        "POTENTIAL_LOCAL_BURNING":
            (
                "Verify local burning activity and inspect "
                "nearby air-quality conditions."
            ),

        "POTENTIAL_DOMESTIC_TRANSPORT":
            (
                "Monitor downwind stations and investigate "
                "regional burning conditions."
            ),

        "POTENTIAL_REGIONAL_TRANSPORT":
            (
                "Monitor downwind stations and coordinate "
                "regional verification across affected areas."
            ),

        "SATELLITE_FIRE_ASSOCIATION":
            (
                "Continue monitoring and seek additional "
                "evidence before source attribution."
            ),
    }

    return mapping.get(
        source_context,
        "Continue monitoring.",
    )


# ============================================================
# BUILD EVENT INTELLIGENCE
# ============================================================

def build_event_intelligence():

    print("\n==============================")
    print("POLLUTION EVENT INTELLIGENCE")
    print("==============================")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    correlation = (
        load_correlation_data()
    )

    landcover = (
        load_landcover_data()
    )

    print(
        "Pollution correlation links:",
        len(correlation),
    )

    print(
        "Land-cover event records:",
        len(landcover),
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    intelligence = correlation.merge(
        landcover,
        on="fire_event_id",
        how="left",
        suffixes=(
            "",
            "_landcover",
        ),
    )

    # --------------------------------------------------------
    # CHECK MATCHING
    # --------------------------------------------------------

    matched = (
        intelligence[
            "classification_status"
        ]
        == "SUCCESS"
    )

    print(
        "Correlation records with land-cover context:",
        int(matched.sum()),
        "/",
        len(intelligence),
    )

    # --------------------------------------------------------
    # FILL MISSING LAND COVER
    # --------------------------------------------------------

    landcover_numeric = [
        "dominant_landcover_fraction",
        "cropland_fraction",
        "forest_fraction",
        "shrubland_fraction",
        "grassland_fraction",
        "builtup_fraction",
        "bare_sparse_fraction",
        "water_fraction",
        "wetland_fraction",
        "mangrove_fraction",
        "moss_lichen_fraction",
        "valid_pixel_count",
        "sample_coverage_fraction",
    ]

    for column in landcover_numeric:

        if column in intelligence.columns:

            intelligence[column] = pd.to_numeric(
                intelligence[column],
                errors="coerce",
            )

    # Non-Indian events do not have this land-cover layer yet.
    intelligence["cropland_fraction"] = (
        intelligence[
            "cropland_fraction"
        ]
        .fillna(0.0)
    )

    intelligence["agricultural_context"] = (
        intelligence[
            "agricultural_context"
        ]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # LAND-COVER YEAR
    # --------------------------------------------------------

    intelligence[
        "landcover_source"
    ] = (
        "ESA WorldCover "
        + str(LANDCOVER_YEAR)
    )

    # --------------------------------------------------------
    # AGRICULTURAL BURNING SCORE
    # --------------------------------------------------------

    intelligence[
        "agricultural_burning_score"
    ] = intelligence.apply(
        lambda row:
        calculate_agricultural_score(
            cropland_fraction=row[
                "cropland_fraction"
            ],
            pm25_elevation_score=row[
                "pm25_elevation_score"
            ],
            event_score=row[
                "event_score"
            ],
            fire_confidence=row[
                "fire_confidence"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # SOURCE CONTEXT
    # --------------------------------------------------------

    intelligence[
        "source_context"
    ] = intelligence.apply(
        classify_source_context,
        axis=1,
    )

    # --------------------------------------------------------
    # EVIDENCE SUMMARY
    # --------------------------------------------------------

    intelligence[
        "evidence_summary"
    ] = intelligence.apply(
        build_evidence_summary,
        axis=1,
    )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    intelligence[
        "recommended_action"
    ] = intelligence[
        "source_context"
    ].apply(
        recommended_action
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    def assign_priority(row):

        source = row[
            "source_context"
        ]

        score = float(
            row[
                "event_score"
            ]
        )

        if (
            source
            == "POTENTIAL_REGIONAL_TRANSPORT"
            and score >= 0.60
        ):

            return "HIGH"

        if (
            source
            == "POTENTIAL_AGRICULTURAL_BURNING"
            and score >= 0.60
        ):

            return "HIGH"

        if score >= 0.65:

            return "HIGH"

        if score >= 0.50:

            return "MODERATE"

        return "LOW"

    intelligence[
        "response_priority"
    ] = intelligence.apply(
        assign_priority,
        axis=1,
    )

    # --------------------------------------------------------
    # UNIQUE EVENT / STATION LINKS
    # --------------------------------------------------------

    intelligence = (
        intelligence
        .sort_values(
            [
                "fire_event_id",
                "station_id",
                "event_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "fire_event_id",
                "station_id",
            ],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # SORT BY PRIORITY + SCORE
    # --------------------------------------------------------

    priority_order = {
        "HIGH": 0,
        "MODERATE": 1,
        "LOW": 2,
    }

    intelligence[
        "_priority_order"
    ] = intelligence[
        "response_priority"
    ].map(
        priority_order
    )

    intelligence = (
        intelligence
        .sort_values(
            [
                "_priority_order",
                "event_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns=[
                "_priority_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # ROUND
    # --------------------------------------------------------

    round_columns = [
        "fire_latitude",
        "fire_longitude",
        "fire_max_frp",
        "distance_km",
        "time_lag_hours",
        "pm25_at_response",
        "pm25_zscore",
        "pm25_elevation_score",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_alignment_score",
        "kinematic_travel_time_hours",
        "transport_time_score",
        "event_score",
        "cropland_fraction",
        "agricultural_burning_score",
        "sample_coverage_fraction",
    ]

    for column in round_columns:

        if column in intelligence.columns:

            intelligence[column] = (
                pd.to_numeric(
                    intelligence[column],
                    errors="coerce",
                )
                .round(3)
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    intelligence.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("EVENT INTELLIGENCE RESULTS")
    print("==============================")

    print(
        "Final event-station links:",
        len(intelligence),
    )

    print(
        "Unique fire events:",
        intelligence[
            "fire_event_id"
        ].nunique(),
    )

    print(
        "Unique monitoring stations:",
        intelligence[
            "station_id"
        ].nunique(),
    )

    print(
        "\nSource context:"
    )

    print(
        intelligence[
            "source_context"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nResponse priority:"
    )

    print(
        intelligence[
            "response_priority"
        ]
        .value_counts()
        .reindex(
            [
                "HIGH",
                "MODERATE",
                "LOW",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nAgricultural context:"
    )

    print(
        intelligence[
            "agricultural_context"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nAgricultural burning score:"
    )

    print(
        intelligence[
            "agricultural_burning_score"
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # TOP EVENTS
    # --------------------------------------------------------

    print(
        "\nTop event intelligence:"
    )

    display_columns = [
        "correlation_id",
        "fire_event_id",
        "station",
        "source_context",
        "response_priority",
        "fire_latitude",
        "fire_longitude",
        "fire_timestamp",
        "distance_km",
        "time_lag_hours",
        "pm25_at_response",
        "pm25_zscore",
        "cropland_fraction",
        "agricultural_burning_score",
        "event_score",
        "evidence_level",
        "evidence_summary",
    ]

    print(
        intelligence[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # SAVE FINAL
    # ========================================================

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nPollution event intelligence completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    build_event_intelligence()