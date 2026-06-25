"""
TODO:
- Implement a NeuralNetworkRegressor wrapper derived from BaseRegressor class.
- For loading data use Hea_data_loader module for now or if anyone wanna improve/change it can do so
- Hyperparameters:
    - hidden_layer_sizes
    - activation
    - learning_rate
    - batch_size
    - epochs
    - optimizer
    - dropout (optional)
- Implement:
    - __init__()
    - fit(X_train, y_train)
    - predict(X)
    - score(X, y)
- Allow reproducible random seed.
- Add early stopping.
- Add save() and load().
- Expose hyperparameters for optimization.
"""