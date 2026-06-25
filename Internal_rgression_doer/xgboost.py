"""
TODO:
- Implement an XGBoostRegressor wrapper derived from BaseRegressor class.
- For loading data use Hea_data_loader module for now or if anyone wanna improve/change it can do so
- Hyperparameters:
    - n_estimators
    - max_depth
    - learning_rate
    - subsample
    - colsample_bytree
    - min_child_weight
    - gamma
- Implement:
    - __init__()
    - fit(X_train, y_train)
    - predict(X)
    - score(X, y)
- Support early stopping.
- Add feature importance extraction.
- Add save() and load().
- Expose hyperparameters for optimization algorithms.
"""