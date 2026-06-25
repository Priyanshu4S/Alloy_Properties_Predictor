from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Base class for hyperparameter optimization algorithms.

    Examples:
        - Genetic Algorithm
        - Bayesian Optimization
        - Grid Search
        - Random Search
    """

    @abstractmethod
    def optimize(self, model, dataloader):
        """
        Optimize the model's hyperparameters.

        Returns:
            best_model
            best_score
            best_hyperparameters
        """
        pass

    @abstractmethod
    def objective_function(self, model, dataloader):
        """
        Evaluate one candidate solution.
        """
        pass