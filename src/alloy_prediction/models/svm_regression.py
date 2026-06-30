"""
Support Vector Regression predictor.

Wraps sklearn.svm.SVR and follows the BasePredictor API.

Use this for regression targets such as:
- Hardness
- Density_calc

Do not use this for categorical targets such as Phases.
For Phases, use SVC instead of SVR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pickle
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from alloy_prediction.models.base_model import BasePredictor


class SVMRegressionPredictor(BasePredictor):
    """
    Support Vector Regression model for alloy property prediction.
    """

    def __init__(
        self,
        target_property: str,
        random_state: int | None = None,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
        **hyperparameters: Any,
    ) -> None:

        all_hyperparameters = {
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma,
            **hyperparameters,
        }

        super().__init__(
            target_property=target_property,
            random_state=random_state,
            **all_hyperparameters,
        )

        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svr", SVR(**self.hyperparameters)),
            ]
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "SVMRegressionPredictor":
        """
        Train the SVR model.
        """

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        y = np.asarray(y)

        try:
            y = y.astype(float)
        except ValueError as exc:
            raise ValueError(
                "SVR is a regression model, so y must be numeric. "
                f"Current target '{self.target_property}' may be categorical."
            ) from exc

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict target values for the given samples.
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        return self.model.predict(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Return R² score for the model.
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        y = np.asarray(y)

        try:
            y = y.astype(float)
        except ValueError as exc:
            raise ValueError(
                "SVR score calculation needs numeric y values."
            ) from exc

        y_pred = self.predict(X)

        return r2_score(y, y_pred)

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained predictor.
        """

        path = Path(path)

        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "SVMRegressionPredictor":
        """
        Load a previously saved predictor.
        """

        path = Path(path)

        with path.open("rb") as file:
            predictor = pickle.load(file)

        if not isinstance(predictor, cls):
            raise TypeError(
                f"Loaded object is not a {cls.__name__}."
            )

        return predictor


hardness_predictor: SVMRegressionPredictor | None = None
density_predictor: SVMRegressionPredictor | None = None


def train_svm_regression_predictor(
    data_loader,
    target_property: str | None = None,
    **model_parameters: Any,
) -> tuple[SVMRegressionPredictor, float]:
    """
    Train an SVM regression predictor using a prepared data loader.

    Returns
    -------
    predictor
        Trained SVMRegressionPredictor.

    score
        R² score on test data.
    """

    X_train, X_test, y_train, y_test = data_loader.get_data()

    if target_property is None:
        target_property = data_loader.get_target_name()

    predictor = SVMRegressionPredictor(
        target_property=target_property,
        **model_parameters,
    )

    predictor.fit(X_train, y_train)

    score = predictor.score(X_test, y_test)

    return predictor, score


def train_named_svm_regression_predictor(
    data_loader,
    **model_parameters: Any,
) -> tuple[SVMRegressionPredictor, float]:
    """
    Train predictor and expose it as hardness_predictor or density_predictor
    depending on the target name.
    """

    global hardness_predictor
    global density_predictor

    predictor, score = train_svm_regression_predictor(
        data_loader,
        **model_parameters,
    )

    target_name = predictor.target_property.lower()

    if "hardness" in target_name:
        hardness_predictor = predictor

    elif "density" in target_name:
        density_predictor = predictor

    return predictor, score