import pandas as pd
from srcs.Mlp import MLP
from srcs.Scaler import Scaler
from srcs.display import plot_learning_curves
    
def training(layer, epochs, loss, batch_size, learning_rate, seed, standardize, weight_initializer):
    X_train = pd.read_csv('data/X_train.csv', header=None)
    y_train = pd.read_csv('data/y_train.csv', header=None).values.ravel()

    X_test = pd.read_csv("data/X_test.csv", header=None)
    y_test = pd.read_csv("data/y_test.csv", header=None).values.ravel()

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_test.shape}")

    scaler = Scaler(method=standardize)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLP(hidden_layer_sizes=layer,
                epochs=epochs,
                loss=loss,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=seed,
                weight_initializer=weight_initializer)
    model.fit(X_train, y_train, X_test, y_test)
    print(model)
    plot_learning_curves(model.train_losses, model.val_losses, model.train_accuracies, model.val_accuracies)