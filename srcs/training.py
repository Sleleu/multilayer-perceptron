import pandas as pd
import numpy as np
from srcs.utils import YELLOW, END
from srcs.Mlp import MLP
from srcs.Scaler import Scaler
from srcs.display import plot_learning_curves
from srcs.EarlyStopping import EarlyStopping
    
def save_model(model, W, b, filepath='saved_model.npy'):
    model_data = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'output_layer_size': model.output_layer_size,
        'activation': model.activation_name,
        'output_activation': model.output_activation_name,
        'loss': model.loss_name,
        'W': W,
        'b': b
    }
    np.save(filepath, model_data)
    print(f"Saving model '{filepath}' to disk...")

def training(layer, epochs, loss, batch_size, 
             learning_rate, seed, standardize, 
             weight_initializer, solver, patience,
             activation, output_activation):
    X_train = pd.read_csv('data/processed/train/X_train.csv', header=None)
    y_train = pd.read_csv('data/processed/train/y_train.csv', header=None).values.ravel()

    X_test = pd.read_csv("data/processed/val/X_val.csv", header=None)
    y_test = pd.read_csv("data/processed/val/y_val.csv", header=None).values.ravel()

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_test.shape}")

    scaler = Scaler(method=standardize)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    scaler_params = {
        'method': standardize,
        'mean': scaler.mean,
        'scale': scaler.scale,
        'min': scaler.min,
        'max': scaler.max
    }
    np.save('scaler_params.npy', scaler_params)

    early_stopping = EarlyStopping(patience=patience)

    model = MLP(hidden_layer_sizes=layer,
                activation=activation,
                output_activation=output_activation,
                epochs=epochs,
                loss=loss,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=seed,
                weight_initializer=weight_initializer,
                solver=solver)
    try:
        best_W, best_b = model.fit(X_train, y_train, X_test, y_test, early_stopping)
        save_model(model, best_W, best_b)
    except ValueError as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        exit(1)
    print(model)
    plot_learning_curves(model.train_losses, model.val_losses, model.train_accuracies, model.val_accuracies)