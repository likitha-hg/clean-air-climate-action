from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .client import ClientUpdate


# ============================================================
# GLOBAL FEDERATED MODEL
# ============================================================

@dataclass
class GlobalModel:
    """
    Aggregated federated model parameters.
    """

    coefficients: np.ndarray
    intercept: float
    total_samples: int
    participating_clients: int


# ============================================================
# FEDERATED AGGREGATOR
# ============================================================

class FederatedAggregator:
    """
    Central coordinator for federated parameter aggregation.

    Only model parameters are aggregated.

    Raw environmental observations remain inside
    each local federated client.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
    ) -> None:

        self.feature_names = list(feature_names)

        if not self.feature_names:
            raise ValueError(
                "feature_names must contain at least one feature."
            )

        self.global_model: GlobalModel | None = None

    # --------------------------------------------------------
    # VALIDATE CLIENT UPDATE
    # --------------------------------------------------------

    def _validate_update(
        self,
        update: ClientUpdate,
    ) -> None:

        coefficients = np.asarray(
            update.coefficients,
            dtype=np.float64,
        )

        if coefficients.ndim != 1:
            raise ValueError(
                f"Client '{update.client_id}' coefficients "
                "must be a 1-dimensional array."
            )

        if len(coefficients) != len(self.feature_names):
            raise ValueError(
                f"Client '{update.client_id}' has "
                f"{len(coefficients)} coefficients; "
                f"expected {len(self.feature_names)}."
            )

        if update.sample_count <= 0:
            raise ValueError(
                f"Client '{update.client_id}' must contain "
                "at least one training sample."
            )

        if not np.isfinite(coefficients).all():
            raise ValueError(
                f"Client '{update.client_id}' coefficients "
                "contain NaN or infinite values."
            )

        if not np.isfinite(update.intercept):
            raise ValueError(
                f"Client '{update.client_id}' intercept is invalid."
            )

    # --------------------------------------------------------
    # SAMPLE-WEIGHTED FEDAVG
    # --------------------------------------------------------

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
    ) -> GlobalModel:

        updates = list(updates)

        if not updates:
            raise ValueError(
                "At least one client update is required."
            )

        for update in updates:
            self._validate_update(update)

        total_samples = sum(
            update.sample_count
            for update in updates
        )

        if total_samples <= 0:
            raise ValueError(
                "Total client sample count must be positive."
            )

        coefficients = np.zeros(
            len(self.feature_names),
            dtype=np.float64,
        )

        intercept = 0.0

        for update in updates:

            weight = (
                update.sample_count
                / total_samples
            )

            coefficients += (
                weight
                * np.asarray(
                    update.coefficients,
                    dtype=np.float64,
                )
            )

            intercept += (
                weight
                * float(update.intercept)
            )

        self.global_model = GlobalModel(
            coefficients=coefficients,
            intercept=float(intercept),
            total_samples=int(total_samples),
            participating_clients=len(updates),
        )

        return self.global_model

    # --------------------------------------------------------
    # GET GLOBAL PARAMETERS
    # --------------------------------------------------------

    def get_global_parameters(
        self,
    ) -> tuple[np.ndarray, float]:

        if self.global_model is None:
            raise RuntimeError(
                "No global model has been aggregated yet."
            )

        return (
            self.global_model.coefficients.copy(),
            float(self.global_model.intercept),
        )

    # --------------------------------------------------------
    # GLOBAL MODEL SUMMARY
    # --------------------------------------------------------

    def summary(self) -> dict[str, int | float]:

        if self.global_model is None:
            return {
                "aggregated": False,
                "participating_clients": 0,
                "total_samples": 0,
                "feature_count": len(self.feature_names),
            }

        return {
            "aggregated": True,
            "participating_clients": (
                self.global_model.participating_clients
            ),
            "total_samples": (
                self.global_model.total_samples
            ),
            "feature_count": len(self.feature_names),
        }