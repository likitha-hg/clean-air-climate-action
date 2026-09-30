"""
Station-to-state mapping for the current India monitoring network.

The mapping is based on the station identity/jurisdiction present
in the monitoring station names and their associated state
pollution-control-board abbreviations.

This mapping is used only to connect existing monitoring stations
to the published geographic scope of industrial/economic corridors.

It does NOT imply that a station physically lies on an official
corridor route or at an official industrial node.
"""

STATION_STATE = {

    # --------------------------------------------------------
    # Jammu & Kashmir
    # --------------------------------------------------------

    "Rajbagh, Srinagar - JKSPCB":
        "Jammu and Kashmir",

    # --------------------------------------------------------
    # Delhi
    # --------------------------------------------------------

    "ITO, New Delhi - CPCB":
        "Delhi",

    # --------------------------------------------------------
    # Punjab
    # --------------------------------------------------------

    "Civil Line, Jalandhar - PPCB":
        "Punjab",

    # --------------------------------------------------------
    # Haryana
    # --------------------------------------------------------

    "Patti Mehar, Ambala - HSPCB":
        "Haryana",

    "Huda Sector, Fatehabad - HSPCB":
        "Haryana",

    # --------------------------------------------------------
    # Uttar Pradesh
    # --------------------------------------------------------

    "Jigar Colony, Moradabad - UPPCB":
        "Uttar Pradesh",

    "Shastripuram, Agra - UPPCB":
        "Uttar Pradesh",

    "Ardhali Bazar, Varanasi - UPPCB":
        "Uttar Pradesh",

    # --------------------------------------------------------
    # Rajasthan
    # --------------------------------------------------------

    "Indra Nagar, Jhunjhunu - RSPCB":
        "Rajasthan",

    "Khatikan Mohalla, Dausa - RSPCB":
        "Rajasthan",

    "Digari Kalan, Jodhpur - RSPCB":
        "Rajasthan",

    "Railway Colony, Barmer - RSPCB":
        "Rajasthan",

    "Ashok Nagar, Udaipur - RSPCB":
        "Rajasthan",

    # --------------------------------------------------------
    # Assam
    # --------------------------------------------------------

    "Girls College, Sivasagar - PCBA":
        "Assam",

    "Central Academy for SFS, Byrnihat - PCBA":
        "Assam",

    # --------------------------------------------------------
    # Bihar
    # --------------------------------------------------------

    "Kamalnath Nagar, Bettiah - BSPCB":
        "Bihar",

    "DRM Office Danapur, Patna - BSPCB":
        "Bihar",

    "Mayaganj, Bhagalpur - BSPCB":
        "Bihar",

    # --------------------------------------------------------
    # West Bengal
    # --------------------------------------------------------

    "Ward-32 Bapupara, Siliguri - WBPCB":
        "West Bengal",

    "Asansol Court Area, Asansol - WBPCB":
        "West Bengal",

    "Victoria, Kolkata - WBPCB":
        "West Bengal",

    # --------------------------------------------------------
    # Chhattisgarh
    # --------------------------------------------------------

    "Nawapara SECL Colony, Chhal - CECB":
        "Chhattisgarh",

    "AIIMS, Raipur - CECB":
        "Chhattisgarh",

    # --------------------------------------------------------
    # Madhya Pradesh
    # --------------------------------------------------------

    "Gole Bazar, Katni - MPPCB":
        "Madhya Pradesh",

    # --------------------------------------------------------
    # Gujarat
    # --------------------------------------------------------

    "Gyaspur, Ahmedabad - IITM":
        "Gujarat",

    # --------------------------------------------------------
    # Odisha
    # --------------------------------------------------------

    "Tata Township, Bileipada - OSPCB":
        "Odisha",

    "CDA Area, Cuttack - OSPCB":
        "Odisha",

    # --------------------------------------------------------
    # Maharashtra
    # --------------------------------------------------------

    "Ambazari, Nagpur - MPCB":
        "Maharashtra",

    "Rachnakar Colony, Aurangabad - MPCB":
        "Maharashtra",

    "Ashta Vinayak Nagar, Hingoli - MPCB":
        "Maharashtra",

    "Kasarvadavali, Thane - MPCB":
        "Maharashtra",

    # --------------------------------------------------------
    # Andhra Pradesh
    # --------------------------------------------------------

    "GVM Corporation, Visakhapatnam - APPCB":
        "Andhra Pradesh",

    "Kanuru, Vijayawada - APPCB":
        "Andhra Pradesh",

    "Tirumala, Tirupati - APPCB":
        "Andhra Pradesh",

    # --------------------------------------------------------
    # Telangana
    # --------------------------------------------------------

    "Kokapet, Hyderabad - TSPCB":
        "Telangana",

    # --------------------------------------------------------
    # Karnataka
    # --------------------------------------------------------

    "Lal Bahadur Shastri Nagar, Kalaburagi - KSPCB":
        "Karnataka",

    "Deshpande Nagar, Hubballi - KSPCB":
        "Karnataka",

    "Hebbal, Bengaluru - KSPCB":
        "Karnataka",

    "Stuart Hill, Madikeri - KSPCB":
        "Karnataka",

    # --------------------------------------------------------
    # Tamil Nadu
    # --------------------------------------------------------

    "Kodungaiyur, Chennai - TNPCB":
        "Tamil Nadu",

    "Kumaran College, Tirupur - TNPCB":
        "Tamil Nadu",

    "Parisutham Nagar, Thanjavur - TNPCB":
        "Tamil Nadu",

    # --------------------------------------------------------
    # Kerala
    # --------------------------------------------------------

    "Polayathode, Kollam - Kerala PCB":
        "Kerala",

    # --------------------------------------------------------
    # Andaman and Nicobar Islands
    # --------------------------------------------------------

    "Police Line, Sri Vijaya Puram - ANPCC":
        "Andaman and Nicobar Islands",
}


def get_station_state(station_name: str) -> str:
    """
    Return the state for a known station.
    """

    if station_name not in STATION_STATE:

        raise KeyError(
            f"Station not found in STATION_STATE: {station_name}"
        )

    return STATION_STATE[station_name]


def list_mapped_stations():
    """
    Return a copy of the station/state mapping.
    """

    return dict(
        STATION_STATE
    )
