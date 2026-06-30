import pandas as pd

from sklearn.model_selection import train_test_split

from alloy_prediction.data.base_data_loader import BaseDataLoader


class HEADataLoader(BaseDataLoader):

    DESCRIPTOR_COLUMNS = [
        "Num_of_Elem",
        "Density_calc",
        "dHmix",
        "dSmix",
        "dGmix",
        "Tm",
        "n.Para",
        "Atom.Size.Diff",
        "Elect.Diff",
        "VEC",
    ]

    NUMERIC_LIKE_COLUMNS = [
        "Homogenization_Temp",
        "Homogenization_Time",
        "Annealing_Temp",
        "Annealing_Time_(min)",
    ]

    def __init__(
        self,
        csv_path,
        target,
        excluded_columns=None,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        encoding = "latin1",
        stratify = False,
    ):

        super().__init__()

        self.csv_path = csv_path
        self.target = target

        self.excluded_columns = excluded_columns or []

        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.encoding = encoding
        self.stratify = stratify

    ##################################
    # Pipeline                       #
    ##################################

    def load_data(self):
        self._data = pd.read_csv(self.csv_path, encoding=self.encoding)

    def preprocess(self):

        self._clean_column_names()
        self._remove_invalid_rows()
        self._clean_units()
        # self._normalize_labels()
        self._remove_unwanted_columns()
        # self._fill_missing_values()
        self._create_descriptors()
        self._encode_categorical_columns()
        self._build_dataset()

        


    def split(self):
        ######################################################################
        stratify_values = None

        if self.stratify and self.shuffle:
            class_counts = self._y.value_counts(dropna=False)

            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify_values = self._y
        ###########ADDED THIS BLOCK, ANY MALFUNTION TRY REMOVING IT##########
        (
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
        ) = train_test_split(
            self._X,
            self._y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_values, 
        )

    ##################################
    # Individual preprocessing steps #
    ##################################

    def _clean_column_names(self):
        """Remove accidental spaces from column names"""
        
        self._data.columns = self._data.columns.str.strip()
        self.excluded_columns = [col.strip() for col in self.excluded_columns]
        self.target = self.target.strip()


    def _remove_invalid_rows(self):
        if self.target not in self._data.columns:
            raise ValueError(f"Target '{self.target}' not found.")

        self._data = self._data.dropna(subset=[self.target])
        self._data = self._data[
            self._data[self.target].astype(str).str.strip() != ""
        ]

    # def _normalize_labels(self):
    #     """Normalize the target labels to a standard format."""
    #     self._data[self.target] = self._data[self.target].str.strip().str.lower()
        

    def _clean_units(self):
        """Convert numeric-looking processing columns into numeric values"""
        for col in self.NUMERIC_LIKE_COLUMNS:
            if col in self._data.columns:
                self._data[col] = self._extract_numeric_value(self._data[col])
            else:
                print(f"Warning: Column '{col}' not found in the dataset.") 

    def _extract_numeric_value(self, series):
        """Extract the first numeric value from strings like '1000?', '1200 C', etc."""

        cleaned = (
            series
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )

        return pd.to_numeric(cleaned, errors="coerce")

    def _remove_unwanted_columns(self):
        """Remove columns that are not suitable as model features.

        Reasons include:
        - Identifier columns
        - Data leakage
        - Extremely sparse columns
        - Inconsistent formatting"""
        columns_to_drop = [col for col in self.excluded_columns if col in self._data.columns and col != self.target]
        self._data = self._data.drop(columns=columns_to_drop,errors="ignore",)
        # First line makes sure that target is not dropped even if it's in excluded_columns.
        
    # def _fill_missing_values(self):
    #     """Fill missing values in the dataset."""
    #     numeric_cols = self._data.select_dtypes(include=["number"]).columns

    #     self._data[numeric_cols] = self._data[numeric_cols].fillna(
    #         self._data[numeric_cols].median()
    #     )

    #     categorical_cols = self._data.select_dtypes(
    #         include=["object", "category"]
    #     ).columns.tolist()

    #     categorical_cols = [col for col in categorical_cols if col != self.target]

    #     self._data[categorical_cols] = self._data[categorical_cols].fillna(
    #         "Unknown"
    #     )

    def _create_descriptors(self):
        """Use existing descriptor columns if present.

        The dataset already contains descriptor columns, so this function
        does not recalculate them.
        """

        existing_descriptors = [
            col for col in self.DESCRIPTOR_COLUMNS
            if col in self._data.columns
        ]

        missing_descriptors = [
            col for col in self.DESCRIPTOR_COLUMNS
            if col not in self._data.columns
        ]

        self._existing_descriptors = existing_descriptors
        self._missing_descriptors = missing_descriptors

    def _encode_categorical_columns(self):
        """Encode categorical columns using one-hot encoding."""
        categorical_cols = self._data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != self.target]
        
        self._data = pd.get_dummies(self._data, columns=categorical_cols, drop_first=True)

    def _build_dataset(self):

        self._X = self._data.drop(columns=[self.target])

        self._y = self._data[self.target]

    ##################################
    # Metadata                       #
    ##################################

    def get_feature_names(self):

        return list(self._X.columns)

    def get_target_name(self):

        return self.target