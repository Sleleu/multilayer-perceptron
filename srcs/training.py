from utils import load
from activation import sigmoid, sigmoid_derivative, relu, relu_derivative
from loss import log_loss
import numpy as np


# ---- Initialisation functions -----

def init_weights(output_size, input_size):
    weights = np.random.rand(output_size, input_size)
    biases = np.random.rand(output_size, 1)
    return weights, biases


def he_uniform(output_size, input_size):
    limit = np.sqrt(6 / input_size)
    weights = np.random.uniform(-limit, limit, (output_size, input_size))
    biases = np.zeros((output_size, 1))
    
    return weights, biases

# ---------------------------------------

def min_max_normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm


def feed_forward(X_train, W, b):
    A = [X_train.T]
    Z = []

    for c in range(len(W)):
        z = np.dot(W[c], A[c]) + b[c] # Z = W.A + b
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
    
    return A

def back_propagate(A, W, y_train):
    m = y_train.shape[1]
    #print(f"m = {m}")
    # m = 500
    dW = [0] * len(W)
    db = [0] * len(W)
    #print(f"dW len = {len(dW)}")    

    dZ = A[-1] - y_train # Derivee derniere couche

    for c in reversed(range(len(W))):
        dW[c] = 1 / m * np.dot(dZ, A[c].T)
        db[c] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        if c > 0:
            dZ = np.dot(W[c].T, dZ) * sigmoid_derivative(A[c])
    
    return dW, db

def update_weights(W, b, dW, db, learning_rate):
    for c in range(len(W)):
        W[c] -= learning_rate * dW[c]
        b[c] -= learning_rate * db[c]
    return W, b

def training(dataset: str, epochs=1000, learning_rate=0.01):
    X_train = load("data/X_train.csv", header=None).to_numpy()
    y_train = load("data/y_train.csv", header=None).to_numpy()
    y_train = y_train.T
    #print(f"shape of X_train: {X_train.shape} | shape of y_train {y_train.shape}")

    print(f"shape of X_train: {X_train.shape} | shape of y_train {y_train.shape}")
    # shape of X_train: (500, 30) | shape of y_train (500, 1)

    # couches et densité par couche
    layers = [X_train.shape[1], 32, 32, 1]
    nb_layers = len(layers)

    # initialiser weights et biases pour chaque couche
    W = []
    b = []
    for i in range(1, nb_layers):
        weights, biases = init_weights(layers[i], layers[i-1])
        W.append(weights)
        b.append(biases)
    
    for c in range(len(W)):
        print(f"shape of W{c + 1}: {W[c].shape} | shape of b{c + 1} {b[c].shape}") 
    # shape of W1: (32, 30) | shape of b1 (32, 1)
    # shape of W2: (32, 32) | shape of b2 (32, 1)
    # shape of W3: (1, 32) | shape of b3 (1, 1)

    A = feed_forward(X_train, W, b)
    for c in range(len(A)):
        print(f"shape of A{c}: {A[c].shape}")
    # shape of A0: (30, 500)
    # shape of A1: (32, 500)
    # shape of A2: (32, 500)
    # shape of A3: (1, 500)

    dW, db = back_propagate(A, W, y_train)
    for c in reversed(range(len(dW))):
        print(f"shape of dW{c + 1}: {dW[c].shape} | shape of db{c + 1}{db[c].shape}")
    #shape of dW3: (1, 32) | shape of db3(1, 1)
    #shape of dW2: (32, 32) | shape of db2(32, 1)
    #shape of dW1: (32, 30) | shape of db1(32, 1)
    for epoch in range(epochs):
        A = feed_forward(X_train, W, b)

        loss = log_loss(y_train, A[-1])
        
        dW, db = back_propagate(A, W, y_train)
        W, b = update_weights(W, b, dW, db, learning_rate)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
