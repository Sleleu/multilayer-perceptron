import numpy as np
import pandas as pd
from loss import sparse_categorical_cross_entropy
from display import plot_learning_curves
from activation import Activation
from Scaler import Scaler

class MLP:
    def __init__(
            self,
            hidden_layer_sizes=(24, 24),
            output_layer_size=2,
            activation="sigmoid",
            output_function="softmax",
            loss="sparseCategoricalCrossentropy",
            learning_rate=0.0314,
            epochs=84,
            batch_size=8,
            ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size
        self.activation = activation
        self.output_function = output_function
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

def feed_forward(X, W, b):
    A = []
    a = X

    for i in range(len(W) - 1):
        z = np.dot(a, W[i]) + b[i]
        a = Activation.sigmoid(z)
        A.append(a)

    z = np.dot(a, W[-1]) + b[-1]
    output = Activation.softmax(z)
    return output, A

def back_propagate(X, y, output, A, W):
    m = X.shape[0]
    dW, db = [], []
    dz = output.copy()
    dz[np.arange(m), y] -= 1
    dz /= m
    
    for i in reversed(range(len(W))):
        a_prev = A[i - 1] if i > 0 else X
        dW_i = np.dot(a_prev.T, dz)
        db_i = np.sum(dz, axis=0, keepdims=True)
        dW.insert(0, dW_i)
        db.insert(0, db_i)

        if i > 0:
            da = np.dot(dz, W[i].T)
            dz = da * (A[i - 1] * (1 - A[i - 1]))
    
    return dW, db

def init_network(layer_sizes: list):
    W = []
    b = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        
        # HE UNIFORM
        weight = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        biase = np.zeros((1, output_size))
        
        W.append(weight)
        b.append(biase)
    
    return W, b

def get_accuracy(X, y, W, b):
    output, _ = feed_forward(X, W, b)
    predictions = np.argmax(output, axis=1)
    return np.mean(predictions == y)

def update_weights(W, b, dW, db, learning_rate):
    for c in range(len(W)):
        W[c] -= learning_rate * dW[c]
        b[c] -= learning_rate * db[c]
    return W, b

def train_network(X_train, y_train, X_test, y_test, learning_rate=0.0314, batch_size=8, epochs=84):
    W, b = init_network([X_train.shape[1], 24, 24, 24, 2])
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            output, A = feed_forward(batch_X, W, b)
            dW, db = back_propagate(batch_X, batch_y, output, A, W)
            W, b = update_weights(W, b, dW, db, learning_rate)

        train_output, _ = feed_forward(X_train, W, b)
        val_output, _ = feed_forward(X_test, W, b)
        
        train_loss = sparse_categorical_cross_entropy(y_train, train_output)
        val_loss = sparse_categorical_cross_entropy(y_test, val_output)
        
        train_accuracy = get_accuracy(X_train, y_train, W, b)
        val_accuracy = get_accuracy(X_test, y_test, W, b)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
              f"acc: {train_accuracy:.4f} - val_acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies, W, b

def training(layer, epochs, loss, batch_size, learning_rate):
    X_train = pd.read_csv('data/X_train.csv', header=None)
    y_train = pd.read_csv('data/y_train.csv', header=None).values.ravel()

    X_test = pd.read_csv("data/X_test.csv", header=None)
    y_test = pd.read_csv("data/y_test.csv", header=None).values.ravel()

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_test.shape}")

    scaler = Scaler(method="z_score")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    train_losses, val_losses, train_accuracies, val_accuracies, W, b = train_network(X_train, y_train, X_test, y_test)
    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)