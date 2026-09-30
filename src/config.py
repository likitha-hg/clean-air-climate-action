import os

from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# ============================================================
# API KEYS
# ============================================================

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config(
    require_firms=False,
):
    """
    Validate required environment variables.

    By default, OpenAQ and Gemini are required.
    FIRMS can be required when running satellite modules.
    """

    missing_keys = []

    if not OPENAQ_API_KEY:
        missing_keys.append("OPENAQ_API_KEY")

    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")

    if require_firms and not FIRMS_MAP_KEY:
        missing_keys.append("FIRMS_MAP_KEY")

    if missing_keys:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing_keys)
        )

    return True