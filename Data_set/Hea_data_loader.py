from sklearn.model_selection import train_test_split
import pandas as pd

from base_data_loader import BaseDataLoader


class HEADataLoader(BaseDataLoader):

    def __init__(self, csv_path):
        super().__init__()

        self.csv_path = csv_path

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)

    def preprocess(self):
        self.data = self.data.dropna()

        self.X = self.data.drop(columns=["Hardness"])

        self.y = self.data["Hardness"]

    def split(self):
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

    def target_name(self):
        return "Hardness"

    def feature_names(self):
        return list(self.X.columns)