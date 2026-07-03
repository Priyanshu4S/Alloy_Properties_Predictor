"""
Feed Forward Neural Network predictor.

Wraps sklearn.neural_network.MLPRegressor and follows the
BasePredictor API.

Use this for regression targets such as:
- Hardness
- Density_calc

Do not use this for categorical targets such as Phases.
For Phases, use MLPClassifier instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pickle
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from alloy_prediction.models.base_model import BasePredictor


class NeuralNetworkPredictor(BasePredictor):
    """
    Feed Forward Neural Network predictor for alloy property prediction.
    """

    def __init__(
        self,
        target_property: str,
        random_state: int |None = 42,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 1000,
        **hyperparameters: Any,
    ) -> None:

        all_hyperparameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            **hyperparameters,
        }

        super().__init__(
            target_property=target_property,
            random_state=random_state,
            **all_hyperparameters,
        )

        self.model = MLPRegressor(
            random_state=random_state,
            **self.hyperparameters,
        )

        self.training_history: dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "NeuralNetworkPredictor":
        """
        Train the neural network.
        """

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        y = np.asarray(y)

        try:
            y = y.astype(float)
        except ValueError as exc:
            raise ValueError(
                "MLPRegressor is a regression model, "
                f"so target '{self.target_property}' must be numeric."
            ) from exc

        self.model.fit(X, y)

        self.training_history["loss_curve"] = self.model.loss_curve_
        self.training_history["n_iter"] = self.model.n_iter_
        self.training_history["best_loss"] = self.model.best_loss_

        self.is_fitted = True

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict target values.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        return self.model.predict(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Return R² score.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before scoring."
            )

        y = np.asarray(y)

        try:
            y = y.astype(float)
        except ValueError as exc:
            raise ValueError(
                "MLPRegressor requires numeric target values."
            ) from exc

        y_pred = self.predict(X)

        return r2_score(y, y_pred)

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save trained predictor.
        """

        path = Path(path)

        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "NeuralNetworkPredictor":
        """
        Load previously saved predictor.
        """

        path = Path(path)

        with path.open("rb") as file:
            predictor = pickle.load(file)

        if not isinstance(
            predictor,
            cls,
        ):
            raise TypeError(
                f"Loaded object is not a {cls.__name__}."
            )

        return predictor

    def get_training_history(
        self,
    ) -> dict[str, Any]:
        """
        Return recorded training history.
        """

        return self.training_history


hardness_predictor: NeuralNetworkPredictor | None = None
density_predictor: NeuralNetworkPredictor | None = None


def train_neural_network_predictor(
    data_loader,
    target_property: str | None = None,
    **model_parameters: Any,
) -> tuple[NeuralNetworkPredictor, float]:
    """
    Train a neural network predictor using a prepared data loader.

    Returns
    -------
    predictor
        Trained NeuralNetworkPredictor.

    score
        R² score on the test data.
    """

    X_train, X_test, y_train, y_test = data_loader.get_data()

    if target_property is None:
        target_property = data_loader.get_target_name()

    predictor = NeuralNetworkPredictor(
        target_property=target_property,
        **model_parameters,
    )

    predictor.fit(X_train, y_train)

    score = predictor.score(
        X_test,
        y_test,
    )

    return predictor, score


def train_named_neural_network_predictor(
    data_loader,
    **model_parameters: Any,
) -> tuple[NeuralNetworkPredictor, float]:
    """
    Train predictor and expose it as hardness_predictor or
    density_predictor depending on the target property.
    """

    global hardness_predictor
    global density_predictor

    predictor, score = train_neural_network_predictor(
        data_loader,
        **model_parameters,
    )

    target_name = predictor.target_property.lower()

    if "hardness" in target_name:
        hardness_predictor = predictor

    elif "density" in target_name:
        density_predictor = predictor

    return predictor, score