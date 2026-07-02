from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import joblib
import numpy as np

from alloy_prediction.models.base_model import BasePredictor


class NeuralNetworkPredictor(BasePredictor):
    """
    Feed Forward Neural Network using sklearn MLPRegressor
    """

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
    ):

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )

        self.random_state = random_state
        self.training_history = {}

    ###################################################
    # BasePredictor API
    ###################################################

    def fit(self, X_train, y_train):

        self.model.fit(X_train, y_train)

        self.training_history["loss_curve"] = self.model.loss_curve_

        self.training_history["n_iter"] = self.model.n_iter_

        return self

    def predict(self, X):

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):

        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            "R2": r2_score(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": rmse,
        }

        self.training_history["metrics"] = metrics

        return metrics

    ###################################################
    # Serialization
    ###################################################

    def save(self, path):

        joblib.dump(
            {
                "model": self.model,
                "history": self.training_history,
            },
            path,
        )

    @classmethod
    def load(cls, path):

        data = joblib.load(path)

        predictor = cls()

        predictor.model = data["model"]
        predictor.training_history = data["history"]

        return predictor

    ###################################################
    # Utilities
    ###################################################

    def get_training_history(self):

        return self.training_history