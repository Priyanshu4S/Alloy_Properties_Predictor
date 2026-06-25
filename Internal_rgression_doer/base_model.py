from abc import ABC, abstractmethod

class BaseRegressor(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def get_hyperparameters(self):
        pass

    @abstractmethod
    def set_hyperparameters(self, params):
        pass