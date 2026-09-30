from pathlib import Path
import json

from PIL import Image
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# GEMINI CLIENT
# ============================================================

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {

        "pollution_visible": {
            "type": "boolean",
            "description": (
                "Whether the image contains visible signs "
                "that may be related to air pollution."
            ),
        },

        "pollution_category": {
            "type": "string",
            "enum": [
                "smoke",
                "dust",
                "haze",
                "vehicle_exhaust",
                "industrial_emission",
                "open_burning",
                "mixed",
                "unclear",
                "none",
            ],
            "description": (
                "Most likely visible pollution-related category."
            ),
        },

        "confidence": {
            "type": "number",
            "description": (
                "Confidence from 0 to 1 based on the visual "
                "evidence and supplied citizen report."
            ),
        },

        "severity": {
            "type": "string",
            "enum": [
                "LOW",
                "MODERATE",
                "HIGH",
                "UNCLEAR",
            ],
            "description": (
                "Visual severity assessment. This is not "
                "a medical or regulatory air-quality classification."
            ),
        },

        "visual_evidence": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": (
                "Short observations directly supported by "
                "the image."
            ),
        },

        "sensor_consistency": {
            "type": "string",
            "enum": [
                "CONSISTENT",
                "HIGHER_THAN_STATION",
                "LOWER_THAN_STATION",
                "NO_STATION_COMPARISON",
                "NO_SENSOR_DATA",
                "UNCLEAR",
            ],
            "description": (
                "Relationship between the supplied citizen PM2.5 "
                "reading and the nearest station PM2.5 reading, "
                "when both are available. This is descriptive only."
            ),
        },

        "sensor_observations": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": (
                "Short observations about supplied citizen sensor "
                "values and their relationship to available context. "
                "Do not invent measurements."
            ),
        },

        "summary": {
            "type": "string",
            "description": (
                "A concise explanation of what the image, report, "
                "sensor values and supplied environmental context indicate."
            ),
        },

        "recommended_action": {
            "type": "string",
            "description": (
                "A practical next action for an authority or "
                "monitoring team. Do not claim a confirmed source."
            ),
        },
    },

    "required": [
        "pollution_visible",
        "pollution_category",
        "confidence",
        "severity",
        "visual_evidence",
        "sensor_consistency",
        "sensor_observations",
        "summary",
        "recommended_action",
    ],
}


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_citizen_report(
    image_path,
    report_text="",
    latitude=None,
    longitude=None,

    # Nearest official / platform monitoring context
    current_pm25=None,
    forecast_pm25_3h=None,
    hotspot_score=None,
    wind_speed=None,

    # Citizen/local sensor context
    citizen_pm25=None,
    citizen_pm10=None,
    citizen_temperature_c=None,
    citizen_humidity_pct=None,
    citizen_sensor_timestamp=None,
    citizen_sensor_source=None,
):
    """
    Analyze a citizen-submitted image, written report and
    optional citizen/local sensor readings using Gemini.

    Gemini provides visual interpretation and contextual reasoning.
    It does NOT replace scientific measurements, calibration,
    regulatory assessment or scientifically validated source attribution.

    Parameters
    ----------
    image_path:
        Path to citizen-submitted image.

    report_text:
        Citizen's written observation.

    latitude, longitude:
        Location associated with the observation.

    current_pm25:
        PM2.5 from the nearest platform monitoring station.

    forecast_pm25_3h:
        Platform three-hour PM2.5 forecast.

    hotspot_score:
        Platform hotspot score for the relevant station.

    wind_speed:
        Available wind-speed context.

    citizen_pm25:
        PM2.5 supplied by the citizen/local sensor.

    citizen_pm10:
        PM10 supplied by the citizen/local sensor.

    citizen_temperature_c:
        Temperature supplied by the citizen/local sensor.

    citizen_humidity_pct:
        Relative humidity supplied by the citizen/local sensor.

    citizen_sensor_timestamp:
        Timestamp supplied with the local sensor reading.

    citizen_sensor_source:
        Source description such as:
        "Local sensor", "Manual entry", "Demo sensor".
    """

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Citizen image not found: {image_path}"
        )


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            image_path
        )

        image.load()

    except Exception as exc:

        raise ValueError(
            f"Could not open image: {exc}"
        ) from exc


    # ========================================================
    # LOCATION CONTEXT
    # ========================================================

    location_context = (
        "Not provided"
    )


    if (
        latitude is not None
        and longitude is not None
    ):

        location_context = (

            f"Latitude: {latitude}, "
            f"Longitude: {longitude}"

        )


    # ========================================================
    # SENSOR VALIDATION HELPERS
    # ========================================================

    def clean_number(
        value,
        label,
    ):

        if value is None:
            return None

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"{label} must be numeric."
            )

        return numeric_value


    citizen_pm25 = clean_number(
        citizen_pm25,
        "citizen_pm25",
    )


    citizen_pm10 = clean_number(
        citizen_pm10,
        "citizen_pm10",
    )


    citizen_temperature_c = clean_number(
        citizen_temperature_c,
        "citizen_temperature_c",
    )


    citizen_humidity_pct = clean_number(
        citizen_humidity_pct,
        "citizen_humidity_pct",
    )


    current_pm25 = clean_number(
        current_pm25,
        "current_pm25",
    )


    forecast_pm25_3h = clean_number(
        forecast_pm25_3h,
        "forecast_pm25_3h",
    )


    hotspot_score = clean_number(
        hotspot_score,
        "hotspot_score",
    )


    wind_speed = clean_number(
        wind_speed,
        "wind_speed",
    )


    # ========================================================
    # BASIC SENSOR RANGE VALIDATION
    # ========================================================

    if citizen_pm25 is not None and citizen_pm25 < 0:

        raise ValueError(
            "Citizen PM2.5 cannot be negative."
        )


    if citizen_pm10 is not None and citizen_pm10 < 0:

        raise ValueError(
            "Citizen PM10 cannot be negative."
        )


    if (
        citizen_humidity_pct is not None
        and not 0 <= citizen_humidity_pct <= 100
    ):

        raise ValueError(
            "Citizen humidity must be between 0 and 100 percent."
        )


    # ========================================================
    # SENSOR CONTEXT
    # ========================================================

    if (
        citizen_pm25 is None
        and citizen_pm10 is None
        and citizen_temperature_c is None
        and citizen_humidity_pct is None
    ):

        citizen_sensor_context = (
            "No citizen/local sensor measurements were provided."
        )

    else:

        citizen_sensor_context = "\n".join(
            [

                (
                    "Citizen PM2.5: "
                    + (
                        f"{citizen_pm25}"
                        if citizen_pm25 is not None
                        else "Not provided"
                    )
                ),

                (
                    "Citizen PM10: "
                    + (
                        f"{citizen_pm10}"
                        if citizen_pm10 is not None
                        else "Not provided"
                    )
                ),

                (
                    "Citizen temperature (°C): "
                    + (
                        f"{citizen_temperature_c}"
                        if citizen_temperature_c is not None
                        else "Not provided"
                    )
                ),

                (
                    "Citizen humidity (%): "
                    + (
                        f"{citizen_humidity_pct}"
                        if citizen_humidity_pct is not None
                        else "Not provided"
                    )
                ),

                (
                    "Citizen sensor timestamp: "
                    + (
                        str(citizen_sensor_timestamp)
                        if citizen_sensor_timestamp
                        else "Not provided"
                    )
                ),

                (
                    "Citizen sensor source: "
                    + (
                        str(citizen_sensor_source)
                        if citizen_sensor_source
                        else "Not provided"
                    )
                ),

            ]
        )


    # ========================================================
    # STATION / PLATFORM CONTEXT
    # ========================================================

    platform_context = "\n".join(
        [

            (
                "Nearest/platform current PM2.5: "
                + (
                    f"{current_pm25}"
                    if current_pm25 is not None
                    else "Not provided"
                )
            ),

            (
                "Platform PM2.5 forecast in 3 hours: "
                + (
                    f"{forecast_pm25_3h}"
                    if forecast_pm25_3h is not None
                    else "Not provided"
                )
            ),

            (
                "Hotspot score: "
                + (
                    f"{hotspot_score}"
                    if hotspot_score is not None
                    else "Not provided"
                )
            ),

            (
                "Wind speed: "
                + (
                    f"{wind_speed}"
                    if wind_speed is not None
                    else "Not provided"
                )
            ),

        ]
    )


    # ========================================================
    # CITIZEN SENSOR VS STATION COMPARISON
    # ========================================================

    sensor_comparison_context = (
        "No citizen-to-station PM2.5 comparison is available."
    )


    if (
        citizen_pm25 is not None
        and current_pm25 is not None
    ):

        difference = (
            citizen_pm25
            - current_pm25
        )

        absolute_difference = abs(
            difference
        )

        if absolute_difference <= 5:

            relationship = (
                "approximately consistent"
            )

        elif difference > 5:

            relationship = (
                "higher than the nearest station"
            )

        else:

            relationship = (
                "lower than the nearest station"
            )

        sensor_comparison_context = (

            "Citizen PM2.5 compared with nearest station PM2.5: "
            f"{difference:+.2f} µg/m³; "
            f"the citizen reading is {relationship}."

        )


    # ========================================================
    # COMPLETE ENVIRONMENTAL CONTEXT
    # ========================================================

    context_text = f"""
Location context:
{location_context}

Citizen/local sensor context:
{citizen_sensor_context}

Platform monitoring context:
{platform_context}

Citizen-to-station sensor comparison:
{sensor_comparison_context}
""".strip()


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are the citizen-intelligence component of an
AI-powered air-quality and climate-resilience platform.

Analyze the supplied citizen photograph together with:

1. The citizen's written report.
2. Any supplied local/citizen sensor measurements.
3. The nearest/platform air-quality monitoring context.
4. The available forecast, hotspot and weather context.

Your task is NOT to prove a pollution source.

Your task is to combine visual and supplied environmental
evidence into a cautious investigation-oriented assessment.

You must:

1. Identify visible pollution-related patterns only when
   supported by the image.

2. Classify the visible pattern into one of the provided
   pollution categories.

3. Give a confidence value between 0 and 1.

4. Describe only visual evidence that can reasonably be observed.

5. Treat citizen sensor measurements as supplied measurements.
   Do not invent, modify or "correct" them.

6. Compare the citizen PM2.5 reading with the nearest/platform
   PM2.5 reading only when both are supplied.

7. Clearly identify whether the citizen PM2.5 appears:
   - approximately consistent,
   - higher,
   - lower,
   - or not comparable with the nearest station.

8. Treat temperature, humidity and other sensor readings as
   contextual measurements, not proof of pollution source.

9. Use station forecasts, hotspot scores and wind information
   as supporting context only.

10. Clearly indicate uncertainty when evidence is ambiguous.

11. Suggest a practical next action such as:
    - verification,
    - additional local monitoring,
    - repeat measurement,
    - satellite/event review,
    - or authority investigation.

Important limitations:

- Do not claim that an image alone proves an industrial emission,
  open-burning event, vehicle emission or other pollution source.

- Do not claim that a citizen sensor reading is scientifically
  validated unless such validation is explicitly supplied.

- Do not infer health diagnoses.

- Do not invent measurements.

- Do not treat model predictions as direct evidence of what
  caused the visible scene.

- Do not infer causality from correlation alone.

- Do not accuse a person, business or organization of causing
  pollution based on the submitted evidence.

Citizen report:
{report_text if report_text else "No written report provided."}

Environmental context:
{context_text}

Return only valid JSON matching the supplied schema.
"""


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                image,
                prompt,
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=RESPONSE_SCHEMA,

            ),
        )

    except Exception as exc:

        raise RuntimeError(
            f"Gemini analysis request failed: {exc}"
        ) from exc


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )


    try:

        result = json.loads(
            response.text
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Gemini returned invalid JSON: "
            f"{response.text}"
        ) from exc


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    required_keys = [

        "pollution_visible",

        "pollution_category",

        "confidence",

        "severity",

        "visual_evidence",

        "sensor_consistency",

        "sensor_observations",

        "summary",

        "recommended_action",

    ]


    missing_keys = [

        key

        for key
        in required_keys

        if key not in result

    ]


    if missing_keys:

        raise ValueError(

            "Gemini response missing fields: "
            f"{missing_keys}"

        )


    # ========================================================
    # VALIDATE RESULT TYPES
    # ========================================================

    result["pollution_visible"] = bool(
        result["pollution_visible"]
    )


    result["pollution_category"] = str(
        result["pollution_category"]
    )


    result["severity"] = str(
        result["severity"]
    )


    result["confidence"] = max(
        0.0,
        min(
            1.0,
            float(
                result["confidence"]
            ),
        ),
    )


    if not isinstance(
        result["visual_evidence"],
        list,
    ):

        result["visual_evidence"] = [
            str(
                result["visual_evidence"]
            )
        ]


    if not isinstance(
        result["sensor_observations"],
        list,
    ):

        result["sensor_observations"] = [
            str(
                result["sensor_observations"]
            )
        ]


    result["summary"] = str(
        result["summary"]
    )


    result["recommended_action"] = str(
        result["recommended_action"]
    )


    result["sensor_consistency"] = str(
        result["sensor_consistency"]
    )


    # ========================================================
    # ADD NON-GENERATIVE SENSOR METADATA
    # ========================================================
    # These values are supplied by the application and are
    # appended directly rather than generated by Gemini.

    result["citizen_sensor"] = {

        "pm25": citizen_pm25,

        "pm10": citizen_pm10,

        "temperature_c":
            citizen_temperature_c,

        "humidity_pct":
            citizen_humidity_pct,

        "timestamp":
            citizen_sensor_timestamp,

        "source":
            citizen_sensor_source,

    }


    # ========================================================
    # ADD DIRECT SENSOR COMPARISON
    # ========================================================
    # Keep this calculation deterministic in Python so the
    # comparison displayed by the UI does not depend on Gemini.

    if (
        citizen_pm25 is not None
        and current_pm25 is not None
    ):

        result["citizen_station_pm25_difference"] = (
            citizen_pm25
            - current_pm25
        )

    else:

        result["citizen_station_pm25_difference"] = None


    return result


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("GEMINI CITIZEN INTELLIGENCE")
    print("==============================")


    print(
        "Gemini client initialized successfully."
    )


    print(
        f"Model: {MODEL_NAME}"
    )


    print(
        "\nThe image-analysis function is ready."
    )


    print(
        "Supported inputs:"
    )


    print(
        " - Citizen photograph"
    )


    print(
        " - Citizen written report"
    )


    print(
        " - Citizen PM2.5"
    )


    print(
        " - Citizen PM10"
    )


    print(
        " - Citizen temperature"
    )


    print(
        " - Citizen humidity"
    )


    print(
        " - Citizen sensor timestamp"
    )


    print(
        " - Citizen sensor source"
    )


    print(
        "\nUse analyze_citizen_report("
        "image_path, ...)"
        " from the Streamlit application."
    )