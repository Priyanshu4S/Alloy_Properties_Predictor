from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Base class for the optimizer finding optimized alloy.

    Examples:
        - Genetic Algorithm
        - Bayesian Optimization
        - Grid Search
        - Random Search
    """
    def __init__(self, predictor, objective):
        self.predictor = predictor
        self.objective = objective
        
    @abstractmethod
    def optimize(self):
        pass
