import numpy as np

class Sgd:
    def __init__(self, learning_rate=0.0314):
        self.name = "SGD"
        self.learning_rate = learning_rate
    
    def update(self, W, b, dW, db):
        for i in range(len(W)):
            W[i] -= self.learning_rate * dW[i]
            b[i] -= self.learning_rate * db[i]
        return W, b
        
class Momentum:
    def __init__(self, learning_rate=0.0314, momentum=0.9):
        self.name = "Momentum"
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vW = None
        self.vb = None

    def update(self, W, b, dW, db):
        if self.vW is None or self.vb is None:
            self.vW = [np.zeros_like(w) for w in W]
            self.vb = [np.zeros_like(bias) for bias in b]
        
        for i in range(len(W)):
            self.vW[i] = self.momentum * self.vW[i] + self.learning_rate * dW[i]
            self.vb[i] = self.momentum * self.vb[i] + self.learning_rate * db[i]
            W[i] -= self.vW[i]
            b[i] -= self.vb[i]
            
        return W, b