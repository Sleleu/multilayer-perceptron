class Sgd:
    def __init__(self, learning_rate=0.0314):
        self.name = "SGD"
        self.learning_rate = learning_rate
    
    def update(self, W, b, dW, db):
        for i in range(len(W)):
            W[i] -= self.learning_rate * dW[i]
            b[i] -= self.learning_rate * db[i]
        return W, b
        
class Adam:
    def __init__(self, learning_rate=0.0314, beta1=0.9, beta2=0.999, epsilon=1e-15):
        self.name = "Adam"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, W, b, dW, db):
        for i in range(len(W)):
            #W[i] -= self.learning_rate * dW[i] # TO CONTINUE WITH ADAM
            b[i] -= self.learning_rate * db[i]
        return W, b