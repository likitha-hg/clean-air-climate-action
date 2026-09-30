import requests

from src.config import OPENAQ_API_KEY


BASE_URL = "https://api.openaq.org/v3"


def get_india_locations(limit=1000):
    """
    Fetch all available OpenAQ monitoring locations in India.
    """

    url = f"{BASE_URL}/locations"

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    params = {
        "iso": "IN",
        "limit": limit,
        "page": 1
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    data = get_india_locations()

    print("OpenAQ connection successful")
    print("Locations returned:", len(data.get("results", [])))
    print("API found:", data.get("meta", {}).get("found"))