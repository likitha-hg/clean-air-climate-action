from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.federated.client import FederatedClient


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FORECAST_FILE = PROCESSED_DIR / "pm25_forecast_dataset.csv"
PRIMARY_MODEL_FILE = MODELS_DIR / "pm25_xgb_model.joblib"

FEDERATED_MODEL_FILE = MODELS_DIR / "federated_pm25_model.joblib"
GLOBAL_METRICS_FILE = PROCESSED_DIR / "federated_training_metrics.csv"
CLIENT_METRICS_FILE = PROCESSED_DIR / "federated_client_metrics.csv"
REGIONAL_METRICS_FILE = PROCESSED_DIR / "federated_regional_evaluation.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80

# We scale the PM2.5 target so that SGD works in a numerically
# comfortable range while predictions are converted back to
# original µg/m³ units for reporting.
TARGET_SCALE = 100.0


# ============================================================
# REGION ASSIGNMENT
# ============================================================

def assign_region(latitude, longitude):
    """
    Technical geographic partitions used to simulate
    independent federated clients.

    These are NOT official administrative boundaries.
    """

    if latitude >= 26:
        return "North India"

    if latitude < 17:
        return "South India"

    if longitude >= 80:
        return "East India"

    if longitude < 76:
        return "West India"

    return "Central India"


def assign_regions_vectorized(df):
    """
    Vectorized region assignment for the full dataframe.
    """

    conditions = [
        df["latitude"] >= 26,
        df["latitude"] < 17,
        (
            (df["longitude"] >= 80)
            & (df["latitude"] >= 17)
            & (df["latitude"] < 26)
        ),
        (
            (df["longitude"] < 76)
            & (df["latitude"] >= 17)
            & (df["latitude"] < 26)
        ),
    ]

    choices = [
        "North India",
        "South India",
        "East India",
        "West India",
    ]

    return np.select(
        conditions,
        choices,
        default="Central India",
    )


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

def find_target_column(df):
    """
    Identify the 3-hour PM2.5 forecast target robustly.
    """

    preferred_names = [
        "target_pm25_3h",
        "pm25_target_3h",
        "pm25_target",
        "target",
        "target_pm25",
        "pm25_t_plus_3h",
        "pm25_t3h",
    ]

    for name in preferred_names:
        if name in df.columns:
            return name

    # Fallback: look for a column containing both PM2.5 and target.
    candidates = [
        column
        for column in df.columns
        if "pm25" in column.lower()
        and "target" in column.lower()
    ]

    if candidates:
        # Prefer a candidate containing 3h.
        three_hour = [
            column
            for column in candidates
            if "3h" in column.lower()
        ]

        if three_hour:
            return three_hour[0]

        return candidates[0]

    raise ValueError(
        "Could not identify the PM2.5 target column. "
        f"Available columns include: {list(df.columns)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("FEDERATED PM2.5 TRAINING")
    print("=" * 60)

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{FORECAST_FILE}"
        )

    if not PRIMARY_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Primary XGBoost model not found:\n{PRIMARY_MODEL_FILE}"
        )

    df = pd.read_csv(
        FORECAST_FILE,
        low_memory=False,
    )

    # --------------------------------------------------------
    # Remove duplicate column names.
    # This prevents pandas from returning a Series instead
    # of a single value when a column such as latitude appears
    # more than once.
    # --------------------------------------------------------

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        print("\nRemoving duplicate columns:")
        print(duplicate_columns)

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ].copy()

    # --------------------------------------------------------
    # Load primary XGBoost model bundle to reuse its exact
    # feature definition.
    # --------------------------------------------------------

    primary_bundle = joblib.load(
        PRIMARY_MODEL_FILE
    )

    if isinstance(primary_bundle, dict):
        primary_model = primary_bundle.get("model")
        feature_names = primary_bundle.get("features")
    else:
        primary_model = primary_bundle
        feature_names = None

    if feature_names is None:
        raise ValueError(
            "Primary model bundle does not contain "
            "'features'."
        )

    feature_names = list(
        dict.fromkeys(feature_names)
    )

    target_column = find_target_column(df)

    # --------------------------------------------------------
    # Required identity / time columns
    # --------------------------------------------------------

    required_identity = [
        "timestamp",
        "latitude",
        "longitude",
    ]

    station_column = None

    if "station_id" in df.columns:
        station_column = "station_id"
    elif "station" in df.columns:
        station_column = "station"

    if station_column is None:
        raise ValueError(
            "Could not find station identifier. "
            "Expected 'station_id' or 'station'."
        )

    required_columns = list(
        dict.fromkeys(
            required_identity
            + [station_column]
            + feature_names
            + [target_column]
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The forecast dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = list(
        dict.fromkeys(
            ["latitude", "longitude"]
            + feature_names
            + [target_column]
        )
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=(
            ["timestamp", "latitude", "longitude"]
            + feature_names
            + [target_column]
        )
    ).copy()

    df = df[
        np.isfinite(
            df[feature_names + [target_column]].to_numpy()
        ).all(axis=1)
    ].copy()

    removed = before - len(df)

    if removed:
        print(
            f"\nRemoved invalid ML rows: {removed:,}"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Usable forecast rows: {len(df):,}"
    )

    print(
        f"Stations: {df[station_column].nunique():,}"
    )

    print(
        f"Features: {len(feature_names)}"
    )

    print("Feature names:")
    print(", ".join(feature_names))

    print(
        f"Target column: {target_column}"
    )

    print(
        f"Time range: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    return (
        df,
        feature_names,
        target_column,
        station_column,
        primary_model,
    )


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def create_time_split(df):
    print("\n" + "=" * 30)
    print("TIME-BASED FEDERATED SPLIT")
    print("=" * 30)

    timestamps = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(timestamps) * TRAIN_FRACTION
    )

    # Protect against edge cases.
    split_index = max(
        1,
        min(
            split_index,
            len(timestamps) - 1,
        ),
    )

    split_timestamp = timestamps.iloc[
        split_index
    ]

    train_df = df[
        df["timestamp"] < split_timestamp
    ].copy()

    test_df = df[
        df["timestamp"] >= split_timestamp
    ].copy()

    print(
        f"Split timestamp: {split_timestamp}"
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    print("Training period:")
    print(
        f"  {train_df['timestamp'].min()} → "
        f"{train_df['timestamp'].max()}"
    )

    print("Testing period:")
    print(
        f"  {test_df['timestamp'].min()} → "
        f"{test_df['timestamp'].max()}"
    )

    return train_df, test_df, split_timestamp


# ============================================================
# FEDERATED AGGREGATION
# ============================================================

def federated_average(client_updates):
    """
    Sample-weighted Federated Averaging (FedAvg).

    Only client model parameters and sample counts are used.
    Raw training observations are not passed here.
    """

    if not client_updates:
        raise ValueError(
            "No client updates available for aggregation."
        )

    total_samples = sum(
        update.sample_count
        for update in client_updates
    )

    if total_samples <= 0:
        raise ValueError(
            "Total client sample count must be positive."
        )

    coefficient_sum = np.zeros_like(
        client_updates[0].coefficients,
        dtype=float,
    )

    intercept_sum = 0.0

    for update in client_updates:
        weight = (
            update.sample_count
            / total_samples
        )

        coefficient_sum += (
            weight
            * update.coefficients
        )

        intercept_sum += (
            weight
            * update.intercept
        )

    return (
        coefficient_sum,
        float(intercept_sum),
        total_samples,
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    (
        df,
        feature_names,
        target_column,
        station_column,
        _primary_model,
    ) = load_data()

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        split_timestamp,
    ) = create_time_split(df)

    # --------------------------------------------------------
    # Region assignment
    # --------------------------------------------------------

    train_df["region"] = assign_regions_vectorized(
        train_df
    )

    test_df["region"] = assign_regions_vectorized(
        test_df
    )

    # --------------------------------------------------------
    # FEDERATED CLIENTS
    #
    # IMPORTANT:
    #
    # There is intentionally NO global StandardScaler here.
    #
    # Each FederatedClient now fits its own scaler using only
    # its own local training rows.
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("FEDERATED CLIENTS")
    print("=" * 30)

    region_order = [
        "Central India",
        "East India",
        "North India",
        "South India",
        "West India",
    ]

    client_data = []

    for region in region_order:

        region_df = train_df[
            train_df["region"] == region
        ].copy()

        if region_df.empty:
            continue

        client_data.append(
            (
                region,
                region_df,
            )
        )

        print(
            f"{region:<18} "
            f"{len(region_df):>6,} rows | "
            f"{region_df[station_column].nunique():>2} stations"
        )

    if not client_data:
        raise RuntimeError(
            "No regional client data was created."
        )

    # --------------------------------------------------------
    # LOCAL CLIENT TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("LOCAL CLIENT TRAINING")
    print("=" * 30)

    client_updates = []
    client_metrics = []

    for index, (
        region,
        region_df,
    ) in enumerate(
        client_data,
        start=1,
    ):

        X_local = region_df[
            feature_names
        ].to_numpy(
            dtype=float
        )

        y_local = region_df[
            target_column
        ].to_numpy(
            dtype=float
        )

        client = FederatedClient(
            client_id=region,
            feature_names=feature_names,
            target_scale=TARGET_SCALE,
            random_state=RANDOM_STATE,
        )

        update = client.train(
            X_local,
            y_local,
        )

        client_updates.append(
            update
        )

        client_metrics.append(
            {
                "client_id": region,
                "samples": update.sample_count,
                "stations": region_df[
                    station_column
                ].nunique(),
                "train_mae_ug_m3": update.train_mae,
                "train_rmse_ug_m3": update.train_rmse,
            }
        )

        print(
            f"[{index}/{len(client_data)}] "
            f"{region:<16} "
            f"samples={update.sample_count:>7,} | "
            f"MAE={update.train_mae:>8.3f} | "
            f"RMSE={update.train_rmse:>8.3f}"
        )

    # --------------------------------------------------------
    # FEDAVG
    #
    # Client parameters have already been converted into a
    # common RAW-FEATURE parameter space inside each client.
    #
    # Therefore the coordinator does not need access to the
    # clients' feature scaling statistics.
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("FEDERATED AGGREGATION")
    print("=" * 30)

    (
        global_coefficients,
        global_intercept,
        total_local_samples,
    ) = federated_average(
        client_updates
    )

    print(
        "Aggregation method: sample-weighted FedAvg"
    )

    print(
        "Parameter space: common raw-feature space"
    )

    print(
        "Client preprocessing: local StandardScaler"
    )

    print(
        f"Participating clients: "
        f"{len(client_updates)}"
    )

    print(
        f"Total local samples: "
        f"{total_local_samples:,}"
    )

    print(
        f"Global feature count: "
        f"{len(global_coefficients)}"
    )

    # --------------------------------------------------------
    # GLOBAL MODEL EVALUATION
    #
    # The aggregated parameters are already in raw-feature
    # space, so test rows DO NOT receive a global scaler.
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("GLOBAL MODEL EVALUATION")
    print("=" * 30)

    X_test = test_df[
        feature_names
    ].to_numpy(
        dtype=float
    )

    y_test = test_df[
        target_column
    ].to_numpy(
        dtype=float
    )

    predicted_scaled = (
        X_test
        @ global_coefficients
        + global_intercept
    )

    # The global parameters were converted back to original
    # PM2.5 units inside each client before aggregation.
    global_predictions = predicted_scaled

    global_mae = mean_absolute_error(
        y_test,
        global_predictions,
    )

    global_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            global_predictions,
        )
    )

    global_r2 = r2_score(
        y_test,
        global_predictions,
    )

    print(
        f"Global test samples: "
        f"{len(test_df):,}"
    )

    print(
        f"Global MAE : "
        f"{global_mae:.3f} µg/m³"
    )

    print(
        f"Global RMSE: "
        f"{global_rmse:.3f} µg/m³"
    )

    print(
        f"Global R²  : "
        f"{global_r2:.3f}"
    )

    # --------------------------------------------------------
    # REGIONAL GLOBAL-MODEL EVALUATION
    # --------------------------------------------------------

    print("\nRegional global-model evaluation:")

    regional_metrics = []

    for region in region_order:

        region_test = test_df[
            test_df["region"] == region
        ].copy()

        if region_test.empty:
            continue

        X_region = region_test[
            feature_names
        ].to_numpy(
            dtype=float
        )

        y_region = region_test[
            target_column
        ].to_numpy(
            dtype=float
        )

        predictions_region = (
            X_region
            @ global_coefficients
            + global_intercept
        )

        regional_mae = mean_absolute_error(
            y_region,
            predictions_region,
        )

        regional_rmse = np.sqrt(
            mean_squared_error(
                y_region,
                predictions_region,
            )
        )

        regional_r2 = r2_score(
            y_region,
            predictions_region,
        )

        station_count = region_test[
            station_column
        ].nunique()

        regional_metrics.append(
            {
                "region": region,
                "samples": len(region_test),
                "stations": station_count,
                "mae_ug_m3": regional_mae,
                "rmse_ug_m3": regional_rmse,
                "r2": regional_r2,
            }
        )

        print(
            f"{region:<16} "
            f"samples={len(region_test):>7,} | "
            f"MAE={regional_mae:>8.3f} | "
            f"RMSE={regional_rmse:>8.3f} | "
            f"R²={regional_r2:>8.3f}"
        )

    # --------------------------------------------------------
    # SAVE FEDERATED MODEL
    # --------------------------------------------------------

    federated_bundle = {
        "model_type":
            "SGDRegressor_FedAvg_RawFeatureSpace",

        "model": {
            "coefficients":
                global_coefficients,

            "intercept":
                float(
                    global_intercept
                ),
        },

        "features":
            feature_names,

        "target_scale":
            TARGET_SCALE,

        "regions": [
            client_id
            for client_id, _ in client_data
        ],

        "aggregation":
            "sample-weighted FedAvg",

        "parameter_space":
            "raw_feature_space",

        "local_preprocessing":
            (
                "Per-client StandardScaler fitted "
                "only on local training rows"
            ),

        "split_timestamp":
            split_timestamp,

        "training_rows":
            len(train_df),

        "testing_rows":
            len(test_df),

        "stations":
            int(
                df[
                    station_column
                ].nunique()
            ),

        "target_column":
            target_column,
    }

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        federated_bundle,
        FEDERATED_MODEL_FILE,
    )

    # --------------------------------------------------------
    # SAVE GLOBAL METRICS
    # --------------------------------------------------------

    global_metrics_df = pd.DataFrame(
        [
            {
                "model":
                    "Federated SGD + FedAvg",

                "test_samples":
                    len(test_df),

                "training_samples":
                    len(train_df),

                "stations":
                    int(
                        df[
                            station_column
                        ].nunique()
                    ),

                "clients":
                    len(client_updates),

                "feature_count":
                    len(feature_names),

                "split_timestamp":
                    split_timestamp,

                "mae_ug_m3":
                    global_mae,

                "rmse_ug_m3":
                    global_rmse,

                "r2":
                    global_r2,

                "aggregation":
                    "sample-weighted FedAvg",

                "parameter_space":
                    "raw_feature_space",

                "local_preprocessing":
                    "per-client StandardScaler",

                "raw_data_transferred":
                    False,
            }
        ]
    )

    global_metrics_df.to_csv(
        GLOBAL_METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE CLIENT METRICS
    # --------------------------------------------------------

    client_metrics_df = pd.DataFrame(
        client_metrics
    )

    client_metrics_df[
        "aggregation_method"
    ] = "sample-weighted FedAvg"

    client_metrics_df[
        "parameter_space"
    ] = "raw_feature_space"

    client_metrics_df[
        "local_preprocessing"
    ] = "per-client StandardScaler"

    client_metrics_df[
        "raw_data_transferred"
    ] = False

    client_metrics_df.to_csv(
        CLIENT_METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE REGIONAL EVALUATION
    # --------------------------------------------------------

    regional_metrics_df = pd.DataFrame(
        regional_metrics
    )

    regional_metrics_df[
        "model"
    ] = "Federated Global Model"

    regional_metrics_df[
        "parameter_space"
    ] = "raw_feature_space"

    regional_metrics_df[
        "split_timestamp"
    ] = split_timestamp

    regional_metrics_df.to_csv(
        REGIONAL_METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\nFederated model saved to:")
    print(FEDERATED_MODEL_FILE)

    print("\nGlobal metrics saved to:")
    print(GLOBAL_METRICS_FILE)

    print("\nClient metrics saved to:")
    print(CLIENT_METRICS_FILE)

    print("\nRegional evaluation saved to:")
    print(REGIONAL_METRICS_FILE)

    # --------------------------------------------------------
    # FEDERATED DATA FLOW SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("FEDERATED DATA FLOW")
    print("=" * 30)

    print(
        f"Local clients: "
        f"{len(client_updates)}"
    )

    print(
        "Local preprocessing:"
    )

    print(
        "  EACH CLIENT FITS ITS OWN StandardScaler"
    )

    print(
        "Raw environmental observations:"
    )

    print(
        "  REMAIN WITHIN THE LOCAL CLIENT PARTITION"
    )

    print(
        "Transferred to aggregator:"
    )

    print(
        "  MODEL PARAMETERS + SAMPLE COUNTS"
    )

    print(
        "Raw X/y data transferred:"
    )

    print(
        "  NO"
    )

    print("\n" + "=" * 60)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
