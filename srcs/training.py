from utils import load
from activation import sigmoid
import numpy as np


def init_weights(output_size, input_size):
    weights = np.random.rand(output_size, input_size)
    biases = np.random.rand(output_size, 1)
    return weights, biases


def feed_forward(X_train, W, b):
    A = [X_train.T]
    Z = []

    for c in range(len(W)):
        z = np.dot(W[c], A[c]) + b[c] # Z = W.A + b
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
    
    return A, Z

def training(dataset: str):
    X_train = load("data/X_train.csv", header=None)
    y_train = load("data/y_train.csv", header=None)
    print(f"shape of X_train: {X_train.shape} | shape of y_train {y_train.shape}")

    # couches et densité par couche
    layers = [X_train.shape[1], 24, 1]
    nb_layers = len(layers)

    # initialiser weights et biases pour chaque couche
    W = []
    b = []
    for i in range(1, nb_layers):
        weights, biases = init_weights(layers[i], layers[i-1])
        W.append(weights)
        b.append(biases)
    
    A, Z = feed_forward(X_train, W, b)
    print(f"A: {A}")
    print(f"Z: {Z[0]}")

