from pathlib import Path
from math import asin, cos, radians, sin, sqrt, atan2, degrees

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRE_EVENTS_FILE = (
    PROJECT_ROOT
    / "data/processed/firms_fire_events_processed.csv"
)

AIR_QUALITY_FILE = (
    PROJECT_ROOT
    / "data/processed/air_quality_clean.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data/raw/weather_training.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fire_pollution_events.csv"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

ANALYSIS_START = pd.Timestamp(
    "2026-09-13",
    tz="UTC",
)

ANALYSIS_END = pd.Timestamp(
    "2026-09-17 21:30:00",
    tz="UTC",
)


# ------------------------------------------------------------
# DOMESTIC FIRE SETTINGS
# ------------------------------------------------------------

# Local/regional Indian fire search radius
DOMESTIC_RADIUS_KM = 120.0

# Maximum period after a domestic fire event in which
# a PM2.5 response is considered relevant.
DOMESTIC_MAX_LAG_HOURS = 18.0


# ------------------------------------------------------------
# REGIONAL / OUTSIDE-INDIA SETTINGS
# ------------------------------------------------------------

# Wider radius for possible regional transport.
REGIONAL_RADIUS_KM = 350.0

# Longer time window for regional transport.
REGIONAL_MAX_LAG_HOURS = 48.0


# ------------------------------------------------------------
# PM2.5 RESPONSE
# ------------------------------------------------------------

# Minimum PM2.5 elevation score required to create
# a fire-pollution association.
MIN_PM25_ELEVATION = 0.20


# ------------------------------------------------------------
# FINAL EVIDENCE THRESHOLD
# ------------------------------------------------------------

MIN_EVENT_SCORE = 0.45


# ------------------------------------------------------------
# TRANSPORT SCREENING
# ------------------------------------------------------------

# Minimum wind speed considered useful for directional
# transport screening.
MIN_TRANSPORT_WIND_MS = 0.5

# For regional transport, observed lag must be at least
# this fraction of the simple kinematic travel time.
#
# This is only a screening rule, NOT a dispersion model.
MIN_TRAVEL_TIME_FRACTION = 0.50


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance between two coordinates.
    """

    earth_radius_km = 6371.0088

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return (
        2
        * earth_radius_km
        * asin(sqrt(a))
    )


# ============================================================
# BEARING
# ============================================================

def bearing_degrees(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Bearing from point 1 to point 2.

    0   = North
    90  = East
    180 = South
    270 = West
    """

    lat1_rad = radians(float(lat1))
    lat2_rad = radians(float(lat2))

    dlon = radians(
        float(lon2) - float(lon1)
    )

    x = (
        sin(dlon)
        * cos(lat2_rad)
    )

    y = (
        cos(lat1_rad)
        * sin(lat2_rad)
        - sin(lat1_rad)
        * cos(lat2_rad)
        * cos(dlon)
    )

    bearing = atan2(
        x,
        y,
    )

    return (
        degrees(bearing)
        + 360.0
    ) % 360.0


# ============================================================
# ANGULAR DIFFERENCE
# ============================================================

def angular_difference(
    first,
    second,
):
    """
    Smallest absolute angular difference.
    """

    difference = abs(
        float(first)
        - float(second)
    )

    return min(
        difference,
        360.0 - difference,
    )


# ============================================================
# WIND TRANSPORT SCORE
# ============================================================

def wind_transport_score(
    fire_to_station_bearing,
    wind_direction_from,
):
    """
    Wind direction is the direction the wind comes FROM.

    Approximate transport direction:
        wind_direction + 180 degrees

    Returns:
        1.0 = strong directional alignment
        0.0 = poor alignment
    """

    if pd.isna(
        wind_direction_from
    ):
        return 0.0

    transport_direction = (
        float(wind_direction_from)
        + 180.0
    ) % 360.0

    difference = angular_difference(
        fire_to_station_bearing,
        transport_direction,
    )

    return max(
        0.0,
        cos(
            radians(difference)
        ),
    )


# ============================================================
# KINEMATIC TRANSPORT SCREENING
# ============================================================

def calculate_transport_plausibility(
    distance_km,
    wind_speed_ms,
    observed_lag_hours,
):
    """
    Calculate a simple wind-based travel-time estimate.

    Formula:
        travel time = distance / wind speed

    IMPORTANT:
    This is not an atmospheric dispersion model.
    It is only a conservative screening layer used to reject
    clearly impossible regional transport relationships.

    Returns:
        travel_time_hours
        transport_time_score
        physically_plausible
    """

    if pd.isna(
        wind_speed_ms
    ):

        return (
            np.nan,
            0.0,
            False,
        )

    wind_speed_ms = float(
        wind_speed_ms
    )

    distance_km = float(
        distance_km
    )

    observed_lag_hours = float(
        observed_lag_hours
    )

    # Extremely weak / calm winds are not reliable
    # for directional transport screening.
    if wind_speed_ms < MIN_TRANSPORT_WIND_MS:

        return (
            np.nan,
            0.0,
            False,
        )

    wind_speed_kmh = (
        wind_speed_ms
        * 3.6
    )

    if wind_speed_kmh <= 0:

        return (
            np.nan,
            0.0,
            False,
        )

    travel_time_hours = (
        distance_km
        / wind_speed_kmh
    )

    minimum_plausible_lag = (
        travel_time_hours
        * MIN_TRAVEL_TIME_FRACTION
    )

    if (
        observed_lag_hours
        < minimum_plausible_lag
    ):

        return (
            travel_time_hours,
            0.0,
            False,
        )

    ratio = (
        observed_lag_hours
        / travel_time_hours
    )

    if ratio <= 1.0:

        score = ratio

    else:

        # Penalize lags much longer than simple travel time,
        # but do not immediately reduce them to zero.
        score = max(
            0.0,
            1.0
            - (
                ratio - 1.0
            ) / 2.0,
        )

    return (
        travel_time_hours,
        float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        ),
        True,
    )


# ============================================================
# LOAD FIRE EVENTS
# ============================================================

def load_fire_events():

    if not FIRE_EVENTS_FILE.exists():

        raise FileNotFoundError(
            f"Fire-event file not found: "
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
        "event_end",
        "detection_count",
        "satellites",
        "max_frp",
        "strongest_confidence",
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
            f"Missing fire-event columns: {missing}"
        )

    # --------------------------------------------------------
    # DATETIME
    # --------------------------------------------------------

    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    events["event_end"] = pd.to_datetime(
        events["event_end"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = [
        "event_latitude",
        "event_longitude",
        "detection_count",
        "max_frp",
        "satellite_strength_score",
    ]

    for column in numeric_columns:

        events[column] = pd.to_numeric(
            events[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # VALID RECORDS
    # --------------------------------------------------------

    events = events.dropna(
        subset=[
            "fire_event_id",
            "event_latitude",
            "event_longitude",
            "event_timestamp",
        ]
    ).copy()

    # --------------------------------------------------------
    # ANALYSIS WINDOW
    # --------------------------------------------------------

    events = events[
        (events["event_timestamp"] >= ANALYSIS_START)
        & (events["event_timestamp"] <= ANALYSIS_END)
    ].copy()

    return (
        events
        .sort_values(
            "event_timestamp"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD AIR QUALITY
# ============================================================

def load_air_quality():

    if not AIR_QUALITY_FILE.exists():

        raise FileNotFoundError(
            f"Air-quality file not found: "
            f"{AIR_QUALITY_FILE}"
        )

    air = pd.read_csv(
        AIR_QUALITY_FILE,
        parse_dates=["timestamp"],
    )

    air["timestamp"] = pd.to_datetime(
        air["timestamp"],
        errors="coerce",
        utc=True,
    )

    air["pm25"] = pd.to_numeric(
        air["pm25"],
        errors="coerce",
    )

    air["latitude"] = pd.to_numeric(
        air["latitude"],
        errors="coerce",
    )

    air["longitude"] = pd.to_numeric(
        air["longitude"],
        errors="coerce",
    )

    air = air.dropna(
        subset=[
            "station_id",
            "timestamp",
            "pm25",
            "latitude",
            "longitude",
        ]
    ).copy()

    return air


# ============================================================
# LOAD WEATHER
# ============================================================

def load_weather():

    if not WEATHER_FILE.exists():

        raise FileNotFoundError(
            f"Weather file not found: "
            f"{WEATHER_FILE}"
        )

    weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["timestamp"],
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
        utc=True,
    )

    required_columns = [
        "station_id",
        "timestamp",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in weather.columns
    ]

    if missing:

        raise ValueError(
            f"Missing weather columns: {missing}"
        )

    weather["wind_speed_10m"] = pd.to_numeric(
        weather["wind_speed_10m"],
        errors="coerce",
    )

    weather["wind_direction_10m"] = pd.to_numeric(
        weather["wind_direction_10m"],
        errors="coerce",
    )

    weather = weather[
        required_columns
    ].drop_duplicates(
        subset=[
            "station_id",
            "timestamp",
        ]
    )

    return weather


# ============================================================
# BUILD STATION BASELINES
# ============================================================

def build_station_baselines(
    air,
):
    """
    Build pre-event station baselines.

    Only observations before the satellite analysis window
    are used to avoid incorporating the event response itself.
    """

    historical = air[
        air["timestamp"] < ANALYSIS_START
    ].copy()

    if historical.empty:

        raise RuntimeError(
            "No pre-event air-quality observations available "
            "for station baselines."
        )

    baseline = (
        historical
        .groupby(
            "station_id"
        )["pm25"]
        .agg(
            baseline_median="median",
            baseline_std="std",
            baseline_p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    baseline["baseline_std"] = (
        baseline["baseline_std"]
        .fillna(1.0)
        .replace(
            0,
            1.0,
        )
    )

    return baseline


# ============================================================
# ATTACH WEATHER
# ============================================================

def attach_weather(
    air,
    weather,
):
    """
    Align station measurements with the corresponding
    hourly weather observation.
    """

    result = air.copy()

    result["weather_timestamp"] = (
        result["timestamp"]
        .dt.floor("h")
    )

    weather = weather.rename(
        columns={
            "timestamp": "weather_timestamp",
        }
    )

    result = result.merge(
        weather,
        on=[
            "station_id",
            "weather_timestamp",
        ],
        how="left",
    )

    return result


# ============================================================
# FIND STATIONS NEAR FIRE EVENT
# ============================================================

def find_candidate_stations(
    fire_event,
    station_metadata,
):
    """
    Find monitoring stations within an appropriate
    search radius.
    """

    if fire_event["fire_scope"] == "INDIA":

        radius_km = DOMESTIC_RADIUS_KM

    else:

        radius_km = REGIONAL_RADIUS_KM

    candidates = []

    for _, station in station_metadata.iterrows():

        distance_km = haversine_km(
            fire_event["event_latitude"],
            fire_event["event_longitude"],
            station["latitude"],
            station["longitude"],
        )

        if distance_km <= radius_km:

            candidates.append(
                (
                    station,
                    distance_km,
                    radius_km,
                )
            )

    return candidates


# ============================================================
# FIND BEST PM2.5 RESPONSE
# ============================================================

def find_best_pm25_response(
    fire_event,
    station_rows,
    baseline,
    max_lag_hours,
):
    """
    Find the strongest PM2.5 response after a fire event.
    """

    event_time = fire_event[
        "event_timestamp"
    ]

    window_end = (
        event_time
        + pd.Timedelta(
            hours=max_lag_hours
        )
    )

    candidates = station_rows[
        (station_rows["timestamp"] >= event_time)
        & (station_rows["timestamp"] <= window_end)
    ].copy()

    if candidates.empty:

        return None

    baseline_median = float(
        baseline["baseline_median"]
    )

    baseline_std = float(
        baseline["baseline_std"]
    )

    baseline_p75 = float(
        baseline["baseline_p75"]
    )

    candidates["pm25_zscore"] = (
        candidates["pm25"]
        - baseline_median
    ) / baseline_std

    candidates["pm25_elevation_score"] = (
        (
            candidates["pm25"]
            - baseline_median
        )
        / (
            3.0
            * baseline_std
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    candidates["above_baseline_p75"] = (
        candidates["pm25"]
        >= baseline_p75
    )

    # Prefer the strongest abnormal response.
    candidates = candidates.sort_values(
        by=[
            "pm25_elevation_score",
            "pm25_zscore",
            "pm25",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return candidates.iloc[0]


# ============================================================
# MAIN CORRELATION ENGINE
# ============================================================

def correlate_events():

    print("\n==============================")
    print("FIRE EVENT → POLLUTION CORRELATION")
    print("==============================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    fire_events = load_fire_events()

    air_all = load_air_quality()

    weather = load_weather()

    baselines = build_station_baselines(
        air_all
    )

    # Analysis-period air quality
    air = air_all[
        (air_all["timestamp"] >= ANALYSIS_START)
        & (air_all["timestamp"] <= ANALYSIS_END)
    ].copy()

    air = attach_weather(
        air,
        weather,
    )

    print(
        "Satellite fire events:",
        len(fire_events),
    )

    print(
        "Monitoring stations:",
        air["station_id"].nunique(),
    )

    # --------------------------------------------------------
    # STATION METADATA
    # --------------------------------------------------------

    station_metadata = (
        air[
            [
                "station_id",
                "station",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(
            "station_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # STATION DATA INDEX
    # --------------------------------------------------------

    station_data = {
        station_id: group.copy()
        for station_id, group
        in air.groupby(
            "station_id"
        )
    }

    baseline_data = {
        station_id: group.iloc[0]
        for station_id, group
        in baselines.groupby(
            "station_id"
        )
    }

    # --------------------------------------------------------
    # CORRELATE FIRE EVENTS
    # --------------------------------------------------------

    results = []

    total_events = len(
        fire_events
    )

    print(
        "\nCorrelating satellite fire events "
        "with monitoring stations..."
    )

    for event_number, (_, fire_event) in enumerate(
        fire_events.iterrows(),
        start=1,
    ):

        # ----------------------------------------------------
        # SEARCH RADIUS
        # ----------------------------------------------------

        if fire_event[
            "fire_scope"
        ] == "INDIA":

            max_lag_hours = (
                DOMESTIC_MAX_LAG_HOURS
            )

        else:

            max_lag_hours = (
                REGIONAL_MAX_LAG_HOURS
            )

        candidates = find_candidate_stations(
            fire_event,
            station_metadata,
        )

        for (
            station,
            distance_km,
            radius_km,
        ) in candidates:

            station_id = station[
                "station_id"
            ]

            if station_id not in station_data:

                continue

            if station_id not in baseline_data:

                continue

            # ------------------------------------------------
            # FIND PM2.5 RESPONSE
            # ------------------------------------------------

            response = find_best_pm25_response(
                fire_event,
                station_data[
                    station_id
                ],
                baseline_data[
                    station_id
                ],
                max_lag_hours,
            )

            if response is None:

                continue

            pm_elevation = float(
                response[
                    "pm25_elevation_score"
                ]
            )

            if (
                pm_elevation
                < MIN_PM25_ELEVATION
            ):

                continue

            # ------------------------------------------------
            # TIME LAG
            # ------------------------------------------------

            lag_hours = (
                response["timestamp"]
                - fire_event[
                    "event_timestamp"
                ]
            ).total_seconds() / 3600.0

            if lag_hours < 0:

                continue

            if lag_hours > max_lag_hours:

                continue

            # ------------------------------------------------
            # SPATIAL SCORE
            # ------------------------------------------------

            spatial_score = max(
                0.0,
                1.0
                - (
                    distance_km
                    / radius_km
                ),
            )

            # ------------------------------------------------
            # TEMPORAL SCORE
            # ------------------------------------------------

            temporal_score = max(
                0.0,
                1.0
                - (
                    lag_hours
                    / max_lag_hours
                ),
            )

            # ------------------------------------------------
            # WEATHER
            # ------------------------------------------------

            wind_speed = response.get(
                "wind_speed_10m",
                np.nan,
            )

            wind_direction = response.get(
                "wind_direction_10m",
                np.nan,
            )

            # ------------------------------------------------
            # WIND DIRECTION
            # ------------------------------------------------

            if pd.isna(
                wind_direction
            ):

                wind_alignment = 0.0

            else:

                fire_to_station_bearing = (
                    bearing_degrees(
                        fire_event[
                            "event_latitude"
                        ],
                        fire_event[
                            "event_longitude"
                        ],
                        station[
                            "latitude"
                        ],
                        station[
                            "longitude"
                        ],
                    )
                )

                wind_alignment = (
                    wind_transport_score(
                        fire_to_station_bearing,
                        wind_direction,
                    )
                )

            # ------------------------------------------------
            # WIND SPEED SUPPORT
            # ------------------------------------------------

            if pd.isna(
                wind_speed
            ):

                wind_speed_support = 0.0

            elif wind_speed < 1.0:

                wind_speed_support = 0.25

            elif wind_speed <= 6.0:

                wind_speed_support = 1.0

            else:

                wind_speed_support = 0.70

            # ------------------------------------------------
            # SIMPLE TRANSPORT SCREENING
            # ------------------------------------------------

            (
                travel_time_hours,
                transport_time_score,
                transport_plausible,
            ) = calculate_transport_plausibility(
                distance_km=distance_km,
                wind_speed_ms=wind_speed,
                observed_lag_hours=lag_hours,
            )

            # ------------------------------------------------
            # LOCAL DOMESTIC EVENT RULE
            # ------------------------------------------------
            #
            # A very local fire should not automatically be
            # rejected because calm winds make the simple
            # distance / wind-speed calculation unsuitable.
            #
            # For longer-distance and outside-India events,
            # transport plausibility is required.
            # ------------------------------------------------

            is_local_domestic = (
                fire_event["fire_scope"]
                == "INDIA"
                and distance_km <= 50.0
            )

            is_regional_event = (
                fire_event["fire_scope"]
                != "INDIA"
                or distance_km > 50.0
            )

            if (
                is_regional_event
                and not transport_plausible
            ):

                continue

            # ------------------------------------------------
            # SATELLITE STRENGTH
            # ------------------------------------------------

            satellite_strength = float(
                fire_event[
                    "satellite_strength_score"
                ]
            )

            # ------------------------------------------------
            # EVENT SCORE
            # ------------------------------------------------

            event_score = (
                0.20 * spatial_score
                + 0.10 * temporal_score
                + 0.30 * pm_elevation
                + 0.15 * wind_alignment
                + 0.15 * transport_time_score
                + 0.10 * satellite_strength
            )

            # Local domestic event adjustment.
            #
            # The absence of physically useful wind transport
            # should not heavily penalize a nearby fire that
            # coincides with a strong local PM2.5 response.
            if is_local_domestic:

                event_score = (
                    0.25 * spatial_score
                    + 0.15 * temporal_score
                    + 0.35 * pm_elevation
                    + 0.10 * wind_alignment
                    + 0.05 * wind_speed_support
                    + 0.10 * satellite_strength
                )

            event_score = float(
                np.clip(
                    event_score,
                    0.0,
                    1.0,
                )
            )

            if event_score < MIN_EVENT_SCORE:

                continue

            # ------------------------------------------------
            # ORIGIN CLASSIFICATION
            # ------------------------------------------------

            if (
                fire_event["fire_scope"]
                == "INDIA"
            ):

                if distance_km <= 50.0:

                    origin_type = (
                        "POTENTIAL_LOCAL_BURNING"
                    )

                else:

                    origin_type = (
                        "POTENTIAL_DOMESTIC_TRANSPORT"
                    )

            elif (
                wind_alignment >= 0.60
                and transport_plausible
            ):

                origin_type = (
                    "POTENTIAL_REGIONAL_TRANSPORT"
                )

            else:

                origin_type = (
                    "SATELLITE_FIRE_ASSOCIATION"
                )

            # ------------------------------------------------
            # EVIDENCE LEVEL
            # ------------------------------------------------

            if event_score >= 0.70:

                evidence_level = (
                    "STRONG"
                )

            elif event_score >= 0.55:

                evidence_level = (
                    "MODERATE"
                )

            else:

                evidence_level = (
                    "LIMITED"
                )

            # ------------------------------------------------
            # REASONS
            # ------------------------------------------------

            reasons = []

            if pm_elevation >= 0.70:

                reasons.append(
                    "strong PM2.5 elevation"
                )

            elif pm_elevation >= 0.40:

                reasons.append(
                    "moderate PM2.5 elevation"
                )

            if distance_km <= 50:

                reasons.append(
                    "fire event is close to station"
                )

            elif distance_km <= 120:

                reasons.append(
                    "fire event is within domestic analysis radius"
                )

            else:

                reasons.append(
                    "fire event is within regional analysis radius"
                )

            if wind_alignment >= 0.70:

                reasons.append(
                    "wind direction strongly supports transport"
                )

            elif wind_alignment >= 0.50:

                reasons.append(
                    "wind direction supports transport"
                )

            if (
                transport_plausible
                and travel_time_hours > 0
            ):

                reasons.append(
                    "observed timing is compatible with "
                    "simple wind-based travel screening"
                )

            if (
                lag_hours <= 2.0
                and pm_elevation >= 0.40
            ):

                reasons.append(
                    "PM2.5 elevation followed fire detection quickly"
                )

            if not reasons:

                reasons.append(
                    "multiple environmental signals align"
                )

            correlation_reason = (
                "; ".join(reasons)
            )

            # ------------------------------------------------
            # RECOMMENDED ACTION
            # ------------------------------------------------

            if origin_type == (
                "POTENTIAL_REGIONAL_TRANSPORT"
            ):

                recommended_action = (
                    "Monitor downwind stations and coordinate "
                    "regional verification."
                )

            elif origin_type == (
                "POTENTIAL_DOMESTIC_TRANSPORT"
            ):

                recommended_action = (
                    "Verify the domestic burning area and "
                    "monitor downwind air-quality conditions."
                )

            elif origin_type == (
                "POTENTIAL_LOCAL_BURNING"
            ):

                recommended_action = (
                    "Verify local burning activity and "
                    "inspect nearby air-quality conditions."
                )

            else:

                recommended_action = (
                    "Continue monitoring and seek additional "
                    "evidence before source attribution."
                )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append(
                {
                    "fire_event_id": fire_event[
                        "fire_event_id"
                    ],

                    "fire_scope": fire_event[
                        "fire_scope"
                    ],

                    "fire_latitude": fire_event[
                        "event_latitude"
                    ],

                    "fire_longitude": fire_event[
                        "event_longitude"
                    ],

                    "fire_timestamp": fire_event[
                        "event_timestamp"
                    ],

                    "fire_event_end": fire_event[
                        "event_end"
                    ],

                    "fire_detection_count": fire_event[
                        "detection_count"
                    ],

                    "fire_satellites": fire_event[
                        "satellites"
                    ],

                    "fire_max_frp": fire_event[
                        "max_frp"
                    ],

                    "fire_confidence": fire_event[
                        "strongest_confidence"
                    ],

                    "station_id": station_id,

                    "station": station[
                        "station"
                    ],

                    "station_latitude": station[
                        "latitude"
                    ],

                    "station_longitude": station[
                        "longitude"
                    ],

                    "station_timestamp": response[
                        "timestamp"
                    ],

                    "distance_km": distance_km,

                    "time_lag_hours": lag_hours,

                    "pm25_at_response": response[
                        "pm25"
                    ],

                    "pm25_baseline_median": baseline_data[
                        station_id
                    ][
                        "baseline_median"
                    ],

                    "pm25_baseline_p75": baseline_data[
                        station_id
                    ][
                        "baseline_p75"
                    ],

                    "pm25_zscore": response[
                        "pm25_zscore"
                    ],

                    "pm25_elevation_score": pm_elevation,

                    "wind_speed_10m": wind_speed,

                    "wind_direction_10m": wind_direction,

                    "wind_alignment_score": wind_alignment,

                    "wind_speed_support": wind_speed_support,

                    "kinematic_travel_time_hours": (
                        travel_time_hours
                    ),

                    "transport_time_score": (
                        transport_time_score
                    ),

                    "transport_plausible": (
                        transport_plausible
                    ),

                    "spatial_score": spatial_score,

                    "temporal_score": temporal_score,

                    "satellite_strength_score": (
                        satellite_strength
                    ),

                    "event_score": event_score,

                    "origin_type": origin_type,

                    "evidence_level": evidence_level,

                    "correlation_reason": (
                        correlation_reason
                    ),

                    "recommended_action": (
                        recommended_action
                    ),
                }
            )

        # ----------------------------------------------------
        # LIGHT PROGRESS OUTPUT
        # ----------------------------------------------------

        if (
            event_number % 100 == 0
            or event_number == total_events
        ):

            print(
                f"Processed {event_number}/{total_events} "
                f"fire events..."
            )

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        print(
            "\nNo fire-associated pollution links met "
            "the configured evidence criteria."
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            f"\nSaved empty result to: {OUTPUT_FILE}"
        )

        return

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # ONE BEST LINK PER FIRE EVENT + STATION
    # --------------------------------------------------------

    results_df = (
        results_df
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
    # SORT BY EVIDENCE
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            "event_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # CORRELATION IDs
    # --------------------------------------------------------

    results_df.insert(
        0,
        "correlation_id",
        [
            f"FPE-{index + 1:04d}"
            for index
            in range(
                len(results_df)
            )
        ],
    )

    # ========================================================
    # ROUND NUMERIC FIELDS
    # ========================================================

    round_columns = [
        "fire_latitude",
        "fire_longitude",
        "station_latitude",
        "station_longitude",
        "distance_km",
        "time_lag_hours",
        "fire_max_frp",
        "pm25_at_response",
        "pm25_baseline_median",
        "pm25_baseline_p75",
        "pm25_zscore",
        "pm25_elevation_score",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_alignment_score",
        "wind_speed_support",
        "kinematic_travel_time_hours",
        "transport_time_score",
        "spatial_score",
        "temporal_score",
        "satellite_strength_score",
        "event_score",
    ]

    for column in round_columns:

        if column in results_df.columns:

            results_df[column] = (
                results_df[column]
                .round(3)
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================")
    print("CORRELATION RESULTS")
    print("==============================")

    print(
        "Fire events analysed:",
        len(fire_events),
    )

    print(
        "Fire-station correlation links:",
        len(results_df),
    )

    print(
        "Unique fire events with evidence:",
        results_df[
            "fire_event_id"
        ].nunique(),
    )

    print(
        "Unique monitoring stations:",
        results_df[
            "station_id"
        ].nunique(),
    )

    # --------------------------------------------------------
    # ORIGIN TYPES
    # --------------------------------------------------------

    print(
        "\nOrigin classification:"
    )

    print(
        results_df[
            "origin_type"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    print(
        "\nEvidence level:"
    )

    print(
        results_df[
            "evidence_level"
        ]
        .value_counts()
        .reindex(
            [
                "STRONG",
                "MODERATE",
                "LIMITED",
            ],
            fill_value=0,
        )
        .to_string()
    )

    # --------------------------------------------------------
    # FIRE SCOPE
    # --------------------------------------------------------

    print(
        "\nFire scope:"
    )

    print(
        results_df[
            "fire_scope"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # TRANSPORT SCREENING
    # --------------------------------------------------------

    print(
        "\nTransport plausibility:"
    )

    print(
        results_df[
            "transport_plausible"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # EVENT SCORE DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nEvent score statistics:"
    )

    print(
        results_df[
            "event_score"
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # TOP EVENTS
    # --------------------------------------------------------

    print(
        "\nTop correlated events:"
    )

    display_columns = [
        "correlation_id",
        "fire_event_id",
        "station",
        "origin_type",
        "fire_latitude",
        "fire_longitude",
        "fire_timestamp",
        "distance_km",
        "time_lag_hours",
        "kinematic_travel_time_hours",
        "pm25_at_response",
        "pm25_zscore",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_alignment_score",
        "transport_time_score",
        "event_score",
        "evidence_level",
        "correlation_reason",
    ]

    print(
        results_df[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nFire-event correlation completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    correlate_events()