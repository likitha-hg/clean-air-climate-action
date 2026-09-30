from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class ClientUpdate:
    """
    Model update produced by one federated client.

    The client performs preprocessing and model training locally.
    Only the converted model parameters, sample count, and local
    training metrics are returned.
    """

    client_id: str
    sample_count: int
    coefficients: np.ndarray
    intercept: float
    train_mae: float
    train_rmse: float


class FederatedClient:
    """
    Simulated federated client.

    Each client:
      1. Keeps its local X/y data locally.
      2. Fits its own StandardScaler on local training data.
      3. Trains an SGDRegressor locally.
      4. Converts the trained parameters into raw-feature space.
      5. Returns only model parameters and summary metadata.
    """

    def __init__(
        self,
        client_id: str,
        feature_names: Sequence[str],
        target_scale: float = 100.0,
        random_state: int = 42,
    ):
        self.client_id = client_id
        self.feature_names = list(feature_names)
        self.target_scale = float(target_scale)
        self.random_state = random_state

    # ========================================================
    # LOCAL PREPROCESSING
    # ========================================================

    @staticmethod
    def _fit_local_scaler(X):
        """
        Fit StandardScaler using ONLY this client's local data.
        """

        scaler = StandardScaler()
        scaler.fit(X)

        feature_mean = np.asarray(
            scaler.mean_,
            dtype=float,
        )

        feature_scale = np.asarray(
            scaler.scale_,
            dtype=float,
        )

        # Prevent division by zero for constant features.
        feature_scale = np.where(
            feature_scale == 0,
            1.0,
            feature_scale,
        )

        return scaler, feature_mean, feature_scale

    @staticmethod
    def _scale_features(
        X,
        feature_mean,
        feature_scale,
    ):
        """
        Standardize local features using the client's own
        local scaler statistics.
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        return (
            X - feature_mean
        ) / feature_scale

    def _scale_target(self, y):
        """
        Scale PM2.5 target for numerically stable SGD training.
        """

        y = np.asarray(
            y,
            dtype=float,
        )

        return y / self.target_scale

    # ========================================================
    # PARAMETER CONVERSION
    # ========================================================

    def _convert_to_raw_feature_space(
        self,
        coefficients,
        intercept,
        feature_mean,
        feature_scale,
    ):
        """
        Convert parameters from the client's standardized
        feature space into raw-feature space.

        Local model:

            y_scaled =
                b + w * ((X - mean) / scale)

        Equivalent raw-space form:

            y =
                b_raw + w_raw * X

        The returned parameters predict PM2.5 directly in
        original µg/m³ units.
        """

        coefficients = np.asarray(
            coefficients,
            dtype=float,
        )

        raw_coefficients = (
            coefficients
            / feature_scale
            * self.target_scale
        )

        raw_intercept = (
            intercept
            - np.sum(
                coefficients
                * feature_mean
                / feature_scale
            )
        )

        raw_intercept *= (
            self.target_scale
        )

        return (
            raw_coefficients,
            float(raw_intercept),
        )

    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def train(
        self,
        X,
        y,
    ) -> ClientUpdate:

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        if X.ndim != 2:

            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if y.ndim != 1:

            raise ValueError(
                "y must be a 1-dimensional array."
            )

        if len(X) != len(y):

            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if X.shape[1] != len(
            self.feature_names
        ):

            raise ValueError(
                "Feature count does not match feature_names: "
                f"X has {X.shape[1]} columns, "
                f"expected {len(self.feature_names)}."
            )

        if len(X) < 2:

            raise ValueError(
                "A client requires at least two training rows."
            )

        if not np.isfinite(X).all():

            raise ValueError(
                "X contains non-finite values."
            )

        if not np.isfinite(y).all():

            raise ValueError(
                "y contains non-finite values."
            )

        # ----------------------------------------------------
        # LOCAL SCALER
        # ----------------------------------------------------

        (
            _local_scaler,
            feature_mean,
            feature_scale,
        ) = self._fit_local_scaler(X)

        X_scaled = self._scale_features(
            X,
            feature_mean,
            feature_scale,
        )

        y_scaled = self._scale_target(
            y
        )

        # ----------------------------------------------------
        # LOCAL MODEL
        # ----------------------------------------------------

        model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=0.001,
            max_iter=2000,
            tol=1e-4,
            learning_rate="adaptive",
            eta0=0.001,
            early_stopping=False,
            average=True,
            shuffle=True,
            random_state=self.random_state,
        )

        model.fit(
            X_scaled,
            y_scaled,
        )

        # ----------------------------------------------------
        # LOCAL TRAINING PREDICTIONS
        # ----------------------------------------------------

        predicted_scaled = model.predict(
            X_scaled
        )

        predictions = (
            predicted_scaled
            * self.target_scale
        )

        actual = np.asarray(
            y,
            dtype=float,
        )

        # ----------------------------------------------------
        # LOCAL METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            actual,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions,
            )
        )

        # ----------------------------------------------------
        # CONVERT PARAMETERS
        #
        # The model was trained using the client's local
        # standardization, so convert it back to raw-feature
        # space before FedAvg aggregation.
        # ----------------------------------------------------

        (
            raw_coefficients,
            raw_intercept,
        ) = self._convert_to_raw_feature_space(
            coefficients=model.coef_,
            intercept=float(
                model.intercept_[0]
            ),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )

        # ----------------------------------------------------
        # RETURN MODEL UPDATE ONLY
        # ----------------------------------------------------

        return ClientUpdate(
            client_id=self.client_id,
            sample_count=len(actual),
            coefficients=raw_coefficients.copy(),
            intercept=float(raw_intercept),
            train_mae=float(mae),
            train_rmse=float(rmse),
        )
