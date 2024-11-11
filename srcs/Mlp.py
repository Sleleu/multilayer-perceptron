import numpy as np
from srcs.WeightInitialiser import WeightInitialiser
from srcs.optimizer import Sgd, Momentum
from srcs.EarlyStopping import EarlyStopping
from srcs.constants import ACTIVATIONS_FUNCTIONS, OUTPUT_ACTIVATIONS, LOSS_FUNCTIONS
from srcs.utils import GREEN, YELLOW, CYAN, MAGENTA, END, get_accuracy

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
            case "momentum":
                self.solver = Momentum(self.learning_rate)
            case _:
                print(f"{YELLOW}Error: Unknow solver '{solver}'."
                      f'Available choices: ["sgd", "momentum"]{END}')
                exit(1)
                
        # Select activation
        if activation not in ACTIVATIONS_FUNCTIONS:
            print(f"{YELLOW}Error: Unknown activation function '{activation}'.\n"
                  f"Available choices: {list(ACTIVATIONS_FUNCTIONS.keys())}{END}")
            exit(1)
        self.activation, self.activation_derivative = ACTIVATIONS_FUNCTIONS[activation]
        self.activation_name = activation
        
        # Select output activation
        if output_activation not in OUTPUT_ACTIVATIONS:
            print(f"{YELLOW}Error: Unknown output activation '{output_activation}'.\n"
                  f"Available choices: {list(OUTPUT_ACTIVATIONS.keys())}{END}")
            exit(1)
        self.output_activation = OUTPUT_ACTIVATIONS[output_activation]
        self.output_activation_name = output_activation

        # Select loss function
        if loss not in LOSS_FUNCTIONS:
            print(f"{YELLOW}Error: Unknow loss function '{loss}'.\n"
                  f"Available choices: {list(LOSS_FUNCTIONS.keys())}{END}")
            exit(1)
        self.loss = LOSS_FUNCTIONS[loss]
        self.loss_name = loss
        
        if loss == "binaryCrossentropy" and self.output_layer_size != 1:
            raise AttributeError("MLP with binaryCrossentropy need 1 output neuron")
        
    def __str__(self):
        separator = f"{CYAN}═{END}" * 50
        output = [f"\n\t{GREEN}MODEL CONFIGURATION:{END}", separator]
        
        architecture = []
        if hasattr(self, 'input_layer_size'):
            architecture.append(str(self.input_layer_size))
        else:
            architecture.append("X features")
        architecture.extend(str(size) for size in self.hidden_layer_sizes)
        architecture.append(str(self.output_layer_size))
        
        main_attrs = {
            'Architecture': f'{YELLOW} → {CYAN}'.join(architecture),
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
            output.append(f"{GREEN}{name:18}: {CYAN}{value}{END}")
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

        if self.loss_name == "binaryCrossentropy":
            dz = (output - y)
        else:
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

    def fit(self, X_train, y_train, X_val = None, y_val = None, # to work : can use fit without val for modularity
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
            val_output, _ = self.feed_forward(X_val, W, b)

            try:
                train_loss = self.loss(y_train, train_output)
                val_loss = self.loss(y_val, val_output)
            except:
                raise ValueError("Invalid loss function for this training model")
            train_accuracy = get_accuracy(train_output, y_train)
            val_accuracy = get_accuracy(val_output, y_val)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            print(f"{MAGENTA}epoch {CYAN}{epoch+1}/{self.epochs}"
                  f"{GREEN} - {MAGENTA}loss: {CYAN}{train_loss:.4f}"
                  f"{GREEN} - {MAGENTA}val_loss: {CYAN}{val_loss:.4f}"
                  f"{GREEN} - {MAGENTA}acc: {CYAN}{train_accuracy:.4f}"
                  f"{GREEN} - {MAGENTA}val_acc: {CYAN}{val_accuracy:.4f}{END}")
            
            if early_stopping is not None:
                early_stopping.check_loss(val_loss)
                if early_stopping.counter == 0:
                    best_W = [w.copy() for w in W]
                    best_b = [b.copy() for b in b]
                    best_epoch = epoch
                if early_stopping.early_stop:
                    print(f"{GREEN}Early stopping triggered. Best epoch was {YELLOW}{best_epoch + 1}{END}")
                    W, b = best_W, best_b
                    break
        return W, b