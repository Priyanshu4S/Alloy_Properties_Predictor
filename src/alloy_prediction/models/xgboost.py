"""
XGBoost predictor implementation.

Implements the BasePredictor interface for regression tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

from alloy_prediction.predictors.base_model import BasePredictor


class XGBoostPredictor(BasePredictor):
    """
    XGBoost implementation of BasePredictor.

    Predicts a single alloy property.
    """

    def __init__(
        self,
        target_property: str,
        random_state: int | None = 42,
        **hyperparameters: Any,
    ) -> None:

        super().__init__(
            target_property=target_property,
            random_state=random_state,
            **hyperparameters,
        )

        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": random_state,
        }

        default_params.update(hyperparameters)

        self.model = XGBRegressor(**default_params)

    ####################################################
    # BasePredictor interface
    ####################################################

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostPredictor":

        self.feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else None
        )

        if X_val is not None and y_val is not None:

            self.model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        else:

            self.model.fit(X, y)

        self.is_fitted = True

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        self._check_is_fitted()

        return self.model.predict(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:

        return r2_score(y, self.predict(X))

    ####################################################
    # Extra evaluation metrics
    ####################################################

    def evaluate(
        self,
        X,
        y,
    ) -> dict[str, float]:

        predictions = self.predict(X)

        mse = mean_squared_error(y, predictions)

        return {
            "R2": r2_score(y, predictions),
            "MAE": mean_absolute_error(y, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
        }

    ####################################################
    # Feature importance
    ####################################################

    def feature_importance(self):

        self._check_is_fitted()

        importance = self.model.feature_importances_

        if self.feature_names is None:
            return importance

        return dict(
            zip(
                self.feature_names,
                importance,
            )
        )

    ####################################################
    # Serialization
    ####################################################

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        payload = {
            "model": self.model,
            "metadata": self.get_metadata(),
        }

        joblib.dump(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "XGBoostPredictor":

        payload = joblib.load(path)

        metadata = payload["metadata"]

        predictor = cls(
            target_property=metadata["target_property"],
            random_state=metadata["random_state"],
            **metadata["hyperparameters"],
        )

        predictor.model = payload["model"]
        predictor.feature_names = metadata["feature_names"]
        predictor.is_fitted = metadata["is_fitted"]

        return predictor

    ####################################################
    # Utilities
    ####################################################

    def _check_is_fitted(self):

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )
