import numpy as np
from srcs.WeightInitialiser import WeightInitialiser
from srcs.optimizer import Sgd, Adam
from srcs.EarlyStopping import EarlyStopping
from srcs.constants import ACTIVATIONS_FUNCTIONS, OUTPUT_ACTIVATIONS, LOSS_FUNCTIONS

class MLP:
    def __init__(
            self, hidden_layer_sizes=[24, 24, 24], output_layer_size=2,
            activation="sigmoid", output_activation="softmax", loss="sparseCategoricalCrossentropy",
            learning_rate=0.0314, epochs=84, batch_size=8, weight_initializer="HeUniform", 
            random_seed=None, solver='sgd',
            ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_initializer = weight_initializer
        self.random_seed = random_seed
        
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        
        # Select solver
        match (solver):
            case "sgd":
                self.solver = Sgd(self.learning_rate)
            case "adam":
                self.solver = Adam(self.learning_rate)
            case _:
                print(f"Error: Unknow solver '{solver}'."
                      'Available choices: ["sgd", "adam"]')
                exit(1)
                
        # Select activation
        if activation not in ACTIVATIONS_FUNCTIONS:
            print(f"Error: Unknown activation function '{activation}'. "
                  f"Available choices: {list(ACTIVATIONS_FUNCTIONS.keys())}")
            exit(1)
        self.activation, self.activation_derivative = ACTIVATIONS_FUNCTIONS[activation]
        self.activation_name = activation
        
        # Select output activation
        if output_activation not in OUTPUT_ACTIVATIONS:
            print(f"Error: Unknown output activation '{output_activation}'. "
                  f"Available choices: {list(OUTPUT_ACTIVATIONS.keys())}")
            exit(1)
        self.output_activation = OUTPUT_ACTIVATIONS[output_activation]
        self.output_activation_name = output_activation

        # Select loss function
        if loss not in LOSS_FUNCTIONS:
            print(f"Error: Unknow loss function '{loss}'."
                  f"Available choices: {list(LOSS_FUNCTIONS.keys())}")
            exit(1)
        self.loss = LOSS_FUNCTIONS[loss]
        self.loss_name = loss
        
    def __str__(self):
        separator = "-" * 50
        output = ["\n\tMODEL CONFIGURATION:", separator]
        
        architecture = []
        if hasattr(self, 'input_layer_size'):
            architecture.append(str(self.input_layer_size))
        else:
            architecture.append("X features")
        architecture.extend(str(size) for size in self.hidden_layer_sizes)
        architecture.append(str(self.output_layer_size))
        
        main_attrs = {
            'Architecture': ' → '.join(architecture),
            'Activation': self.activation_name,
            'Output Activation': self.output_activation_name,
            'Loss Function': self.loss_name,
            'Learning Rate': self.learning_rate,
            'Epochs': self.epochs,
            'Batch Size': self.batch_size,
            'Seed': self.random_seed,
            'Solver': self.solver.name,
        }
        for name, value in main_attrs.items():
            output.append(f"{name:18}: {value}")
        
        if self.train_losses:
            output.extend([
                "",
                "TRAINING METRICS:",
                f"{'Train Loss':15}: first={self.train_losses[0]:.4f}, last={self.train_losses[-1]:.4f}",
                f"{'Val Loss':15}: first={self.val_losses[0]:.4f}, last={self.val_losses[-1]:.4f}",
                f"{'Train Accuracy':15}: first={self.train_accuracies[0]:.4f}, last={self.train_accuracies[-1]:.4f}",
                f"{'Val Accuracy':15}: first={self.val_accuracies[0]:.4f}, last={self.val_accuracies[-1]:.4f}"
            ])
        
        output.append(separator)
        return "\n".join(output)
        
    
    def feed_forward(self, X, W, b):
        A = []
        a = X

        for i in range(len(W) - 1):
            z = np.dot(a, W[i]) + b[i]
            a = self.activation(z)
            A.append(a)

        z = np.dot(a, W[-1]) + b[-1]
        output = self.output_activation(z)
        return output, A

    def back_propagate(self, X, y, output, A, W):
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
                dz = da * self.activation_derivative(A[i - 1])
        
        return dW, db

    def init_network(self, layer_sizes):
        W = []
        b = []
        
        # Random seed, None if not defined
        np.random.seed(self.random_seed)
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            biase = np.zeros((1, output_size))

            # Weight initialisation
            match (self.weight_initializer):
                case "HeNormal":
                    weight = WeightInitialiser.he_normal(input_size, output_size)
                case "HeUniform":
                    weight = WeightInitialiser.he_uniform(input_size, output_size)
                case "GlorotNormal":
                    weight = WeightInitialiser.glorot_normal(input_size, output_size)
                case "GlorotUniform":
                    weight = WeightInitialiser.glorot_uniform(input_size, output_size)
                case _:
                    print("Error while initialise weights")
                    exit(1)
            
            W.append(weight)
            b.append(biase)
        
        return W, b

    def get_accuracy(self, X, y, W, b):
        output, _ = self.feed_forward(X, W, b)
        predictions = np.argmax(output, axis=1)
        return np.mean(predictions == y)

    def fit(self, X_train, y_train, X_test = None, y_test = None, # to work : can use fit without val for modularity
            early_stopping: EarlyStopping = None):
        self.input_layer_size = X_train.shape[1]
        layer_sizes = [self.input_layer_size] + self.hidden_layer_sizes + [self.output_layer_size]
        W, b = self.init_network(layer_sizes)
        
        best_W, best_b = None, None
        best_epoch = 0
        
        for epoch in range(self.epochs):
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                output, A = self.feed_forward(batch_X, W, b)
                dW, db = self.back_propagate(batch_X, batch_y, output, A, W)
                W, b = self.solver.update(W, b, dW, db)

            train_output, _ = self.feed_forward(X_train, W, b)
            val_output, _ = self.feed_forward(X_test, W, b)
            
            try:
                train_loss = self.loss(y_train, train_output)
                val_loss = self.loss(y_test, val_output)
            except:
                raise ValueError("Invalid loss function for this training model")
            train_accuracy = self.get_accuracy(X_train, y_train, W, b)
            val_accuracy = self.get_accuracy(X_test, y_test, W, b)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            print(f"epoch {epoch+1}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                f"acc: {train_accuracy:.4f} - val_acc: {val_accuracy:.4f}")
            
            if early_stopping is not None:
                early_stopping.check_loss(val_loss)
                if early_stopping.counter == 0:
                    best_W = [w.copy() for w in W]
                    best_b = [b.copy() for b in b]
                    best_epoch = epoch
                if early_stopping.early_stop:
                    print(f"Early stopping triggered. Best epoch was {best_epoch + 1}")
                    W, b = best_W, best_b
                    break
        return W, b