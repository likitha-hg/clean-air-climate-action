"""
Indian industrial/economic corridor registry.

The corridor state scopes are based on official Government of
India / NICDC corridor descriptions.

IMPORTANT:
A state-scope association means a monitoring station is inside
the geographic state scope used by the prototype. It does NOT
mean the station physically lies on the corridor route or at an
official industrial node.

The prototype uses:
    DMIC
    AKIC
    CBIC
    VCIC
    ECEC
    BMIC
"""

CORRIDORS = {

    # ========================================================
    # DELHI–MUMBAI INDUSTRIAL CORRIDOR
    # ========================================================

    "DMIC": {
        "name":
            "Delhi–Mumbai Industrial Corridor",

        "short_name":
            "DMIC",

        "type":
            "Industrial Corridor",

        "states": [
            "Uttar Pradesh",
            "Haryana",
            "Madhya Pradesh",
            "Rajasthan",
            "Gujarat",
            "Maharashtra",
        ],
    },


    # ========================================================
    # AMRITSAR–KOLKATA INDUSTRIAL CORRIDOR
    # ========================================================

    "AKIC": {
        "name":
            "Amritsar–Kolkata Industrial Corridor",

        "short_name":
            "AKIC",

        "type":
            "Industrial Corridor",

        "states": [
            "Punjab",
            "Haryana",
            "Uttar Pradesh",
            "Uttarakhand",
            "Bihar",
            "Jharkhand",
            "West Bengal",
        ],
    },


    # ========================================================
    # CHENNAI–BENGALURU INDUSTRIAL CORRIDOR
    # ========================================================

    "CBIC": {
        "name":
            "Chennai–Bengaluru Industrial Corridor",

        "short_name":
            "CBIC",

        "type":
            "Industrial Corridor",

        "states": [
            "Tamil Nadu",
            "Karnataka",
            "Andhra Pradesh",
        ],
    },


    # ========================================================
    # VIZAG–CHENNAI INDUSTRIAL CORRIDOR
    # ========================================================

    "VCIC": {
        "name":
            "Vizag–Chennai Industrial Corridor",

        "short_name":
            "VCIC",

        "type":
            "Industrial Corridor",

        "states": [
            "Andhra Pradesh",
            "Tamil Nadu",
        ],
    },


    # ========================================================
    # EAST COAST ECONOMIC CORRIDOR
    # ========================================================

    "ECEC": {
        "name":
            "East Coast Economic Corridor",

        "short_name":
            "ECEC",

        "type":
            "Economic Corridor",

        "states": [
            "Andhra Pradesh",
            "Odisha",
            "Tamil Nadu",
            "West Bengal",
        ],
    },


    # ========================================================
    # BENGALURU–MUMBAI INDUSTRIAL CORRIDOR
    # ========================================================

    "BMIC": {
        "name":
            "Bengaluru–Mumbai Industrial Corridor",

        "short_name":
            "BMIC",

        "type":
            "Industrial Corridor",

        "states": [
            "Karnataka",
            "Maharashtra",
        ],
    },
}


def get_corridor(
    corridor_id: str,
) -> dict:
    """
    Return one corridor definition.
    """

    if corridor_id not in CORRIDORS:

        raise KeyError(
            f"Unknown corridor: {corridor_id}"
        )

    return CORRIDORS[corridor_id]


def list_corridors():
    """
    Return all corridor definitions
    in registry order.
    """

    return list(
        CORRIDORS.values()
    )
