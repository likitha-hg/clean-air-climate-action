from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/air_quality_ml.csv"
)

CLEAN_OUTPUT = Path(
    "data/processed/air_quality_clean.csv"
)

ANOMALY_OUTPUT = Path(
    "data/processed/air_quality_anomalies.csv"
)


POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


# These are QA screening thresholds.
# They are NOT health or regulatory standards.
SCREENING_LIMITS = {
    "pm25": 1000,
    "pm10": 2000,
    "no2": 1000,
    "so2": 1000,
    "co": 10000,
    "o3": 500,
}


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    print("Original rows:", len(df))

    # -----------------------------------------
    # Identify suspicious records
    # -----------------------------------------

    suspicious_mask = pd.Series(
        False,
        index=df.index
    )

    anomaly_reasons = pd.Series(
        "",
        index=df.index,
        dtype="object"
    )

    for pollutant in POLLUTANTS:

        negative = df[pollutant] < 0

        high = (
            df[pollutant]
            > SCREENING_LIMITS[pollutant]
        )

        # Record reason for audit trail
        for idx in df.index[
            negative | high
        ]:

            reasons = []

            if negative.loc[idx]:
                reasons.append(
                    f"{pollutant}_negative"
                )

            if high.loc[idx]:
                reasons.append(
                    f"{pollutant}_extreme"
                )

            existing = anomaly_reasons.loc[idx]

            if existing:
                anomaly_reasons.loc[idx] = (
                    existing + ";" +
                    ";".join(reasons)
                )
            else:
                anomaly_reasons.loc[idx] = (
                    ";".join(reasons)
                )

        suspicious_mask |= negative | high

    # -----------------------------------------
    # Additional physical consistency check
    # PM2.5 should not exceed PM10.
    # -----------------------------------------

    pm_relationship_issue = (
        df["pm25"] > df["pm10"]
    )

    for idx in df.index[
        pm_relationship_issue
    ]:

        existing = anomaly_reasons.loc[idx]

        if existing:
            anomaly_reasons.loc[idx] = (
                existing +
                ";pm25_greater_than_pm10"
            )
        else:
            anomaly_reasons.loc[idx] = (
                "pm25_greater_than_pm10"
            )

    suspicious_mask |= pm_relationship_issue

    # -----------------------------------------
    # Split clean data and anomalies
    # -----------------------------------------

    anomalies = df[
        suspicious_mask
    ].copy()

    anomalies["anomaly_reason"] = (
        anomaly_reasons.loc[
            anomalies.index
        ]
        .values
    )

    clean = df[
        ~suspicious_mask
    ].copy()

    # -----------------------------------------
    # Save both datasets
    # -----------------------------------------

    CLEAN_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean.to_csv(
        CLEAN_OUTPUT,
        index=False
    )

    anomalies.to_csv(
        ANOMALY_OUTPUT,
        index=False
    )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print("\n==============================")
    print("CLEANING COMPLETE")
    print("==============================")

    print(
        "Original rows:",
        len(df)
    )

    print(
        "Anomaly rows:",
        len(anomalies)
    )

    print(
        "Clean rows:",
        len(clean)
    )

    print(
        "Rows removed:",
        len(df) - len(clean)
    )

    print(
        "Percentage removed:",
        round(
            (len(anomalies) / len(df)) * 100,
            4
        ),
        "%"
    )

    print("\nAnomaly reasons:")

    print(
        anomalies["anomaly_reason"]
        .value_counts()
        .to_string()
    )

    print(
        "\nClean data saved to:",
        CLEAN_OUTPUT
    )

    print(
        "Anomalies saved to:",
        ANOMALY_OUTPUT
    )


if __name__ == "__main__":
    main()