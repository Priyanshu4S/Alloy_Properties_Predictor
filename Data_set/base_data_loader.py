from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """
    Base class for all dataset loaders.

    Responsibilities:
    - Load raw dataset
    - Preprocess dataset
    - Split dataset
    - Return train/test sets
    """

    @abstractmethod
    def load_data(self):
        """
        Load dataset from source.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Perform preprocessing.
        Example:
        - Missing values
        - Encoding
        - Scaling
        """
        pass

    @abstractmethod
    def train_test_split(self):
        """
        Return
            X_train
            X_test
            y_train
            y_test
        """
        pass

    @abstractmethod
    def get_feature_names(self):
        """
        Return list of feature names.
        """
        pass