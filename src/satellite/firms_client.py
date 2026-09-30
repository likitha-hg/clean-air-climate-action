from io import StringIO

import pandas as pd
import requests

from src.config import FIRMS_MAP_KEY, validate_config


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
)

# India bounding box:
# west, south, east, north
INDIA_BBOX = "67,6,98,38"

DEFAULT_SOURCE = "VIIRS_NOAA20_NRT"


# ============================================================
# DOWNLOAD FIRMS FIRE DATA
# ============================================================

def get_firms_fire_data(
    source=DEFAULT_SOURCE,
    bbox=INDIA_BBOX,
    day_range=1,
    date=None,
):
    """
    Download NASA FIRMS active-fire detections.

    Parameters
    ----------
    source : str
        FIRMS satellite product.
        Examples:
        - VIIRS_NOAA20_NRT
        - VIIRS_NOAA21_NRT

    bbox : str
        west,south,east,north

    day_range : int
        Number of days to retrieve.
        FIRMS Area API supports 1-5 days.

    date : str or None
        YYYY-MM-DD for a historical window.
        None retrieves the most recent available data.

    Returns
    -------
    pandas.DataFrame
        FIRMS fire detections.
    """

    validate_config(require_firms=True)

    if day_range < 1 or day_range > 5:
        raise ValueError(
            "day_range must be between 1 and 5."
        )

    # --------------------------------------------------------
    # BUILD URL
    # --------------------------------------------------------

    url = (
        f"{BASE_URL}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{bbox}/"
        f"{day_range}"
    )

    if date:
        url = f"{url}/{date}"

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    print("Requesting NASA FIRMS data...")
    print(f"Source: {source}")
    print(f"Bounding box: {bbox}")
    print(f"Day range: {day_range}")

    if date:
        print(f"Start date: {date}")
    else:
        print("Period: Most recent available data")

    response = requests.get(
        url,
        timeout=60,
    )

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    if response.status_code != 200:

        raise RuntimeError(
            "NASA FIRMS request failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text[:500]}"
        )

    if not response.text.strip():

        return pd.DataFrame()

    # --------------------------------------------------------
    # PARSE CSV
    # --------------------------------------------------------

    try:

        fire_data = pd.read_csv(
            StringIO(response.text)
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not parse FIRMS CSV response: {exc}"
        ) from exc

    return fire_data


# ============================================================
# SIMPLE VALIDATION
# ============================================================

def validate_fire_data(df):
    """
    Check that the downloaded FIRMS dataset contains
    the key fields required by the event-detection pipeline.
    """

    required_columns = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "frp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"FIRMS data is missing required columns: {missing}"
        )

    return True


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("NASA FIRMS SATELLITE TEST")
    print("==============================")

    fire_data = get_firms_fire_data(
        source="VIIRS_NOAA20_NRT",
        bbox=INDIA_BBOX,
        day_range=1,
    )

    print(
        "\nFire detections returned:",
        len(fire_data)
    )

    if not fire_data.empty:

        validate_fire_data(fire_data)

        print(
            "\nColumns returned:"
        )

        print(
            fire_data.columns.tolist()
        )

        print(
            "\nSample detections:"
        )

        print(
            fire_data.head(10).to_string(index=False)
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
            "\nFIRMS satellite test successful."
        )

    else:

        print(
            "\nNo fire detections were returned "
            "for the requested period."
        )

        print(
            "The FIRMS connection itself was successful."
        )