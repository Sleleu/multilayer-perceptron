import numpy as np
import matplotlib.pyplot as plt

class Activation:

    @staticmethod
    def sigmoid(Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(A):
        return A * (1 - A)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0) * 1
    
    @staticmethod
    def leaky_relu(Z, alpha=0.1):
        Z = np.clip(Z, -500, 500)
        return np.where(Z > 0, Z, alpha * Z)

    @staticmethod
    def leaky_relu_derivative(Z, alpha=0.1):
        return np.where(Z > 0, 1, alpha)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_derivative(A):
        return 1 - A ** 2

    @staticmethod
    def softmax(Z):
        assert len(Z.shape) == 2
        Z_max = np.max(Z, axis=1, keepdims=1)
        e_x = np.exp(Z - Z_max)
        div = np.sum(e_x, axis=1, keepdims=1)
        return e_x / div

def activation_tests():
    print("=== SOFTMAX TESTS ===")
    Z = np.array([[1, 3, 2.5, 5, 4, 2]])
    A = Activation.softmax(Z)
    print("--- Test from wikipedia ---")
    print(f"A: {A}")
    print(f"Sum of A: {np.sum(A)}")

    Z = np.array([[1, 2, 3, 6],  # sample 1
               [2, 4, 5, 6],  # sample 2
               [1, 2, 3, 6]]) # sample 1 again(!)
    A = Activation.softmax(Z)
    print("\n--- Test from stackoverflow, broadcasting ---")
    print(f"A: {A}")
    print(f"Sum of A: {np.sum(A, axis=1)}")

    print("\n--- Sigmoid ---")
    Z = np.array([-1, 0, 1])
    A = Activation.sigmoid(Z)
    print(f"Input: {Z}, Sigmoid Output: {A}")

    print("\n--- Sigmoid Derivative ---")
    dA = Activation.sigmoid_derivative(A)
    print(f"Sigmoid Derivative Output: {dA}")

    print("\n--- ReLU ---")
    Z = np.array([-1, 0, 1])
    A = Activation.relu(Z)
    print(f"Input: {Z}, ReLU Output: {A}")

    print("\n--- ReLU Derivative ---")
    dA = Activation.relu_derivative(Z)
    print(f"ReLU Derivative Output: {dA}")

    print("\n--- Leaky ReLU ---")
    Z = np.array([-1, 0, 1])
    A = Activation.leaky_relu(Z)
    print(f"Input: {Z}, Leaky ReLU Output: {A}")

    print("\n--- Leaky ReLU Derivative ---")
    dA = Activation.leaky_relu_derivative(Z)
    print(f"Leaky ReLU Derivative Output: {dA}")

    print("\n--- Tanh ---")
    Z = np.array([-1, 0, 1])
    A = Activation.tanh(Z)
    print(f"Input: {Z}, Tanh Output: {A}")

    print("\n--- Tanh Derivative ---")
    dA = Activation.tanh_derivative(A)
    print(f"Tanh Derivative Output: {dA}")

def plot_activation_functions():
    Z = np.linspace(-10, 10, 100)
    
    activations = {
        "Sigmoid": Activation.sigmoid(Z),
        "ReLU": Activation.relu(Z),
        "Leaky ReLU": Activation.leaky_relu(Z),
        "Tanh": Activation.tanh(Z)
    }
    derivatives = {
        "Sigmoid": Activation.sigmoid_derivative(Activation.sigmoid(Z)),
        "ReLU": Activation.relu_derivative(Z),
        "Leaky ReLU": Activation.leaky_relu_derivative(Z),
        "Tanh": Activation.tanh_derivative(Activation.tanh(Z))
    }
    colors_activation = ["blue", "red", "orange", "lime"]
    colors_derivative = ["darkviolet", "darkred", "sandybrown", "green"]
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    for i, (name, values) in enumerate(activations.items()):
        row, col = divmod(i, 2)
        
        axes[row, col].plot(Z, values, label=f"{name}", color=colors_activation[i])
        axes[row, col].plot(Z, derivatives[name], label=f"{name} Derivative", color=colors_derivative[i], linestyle="--")
        axes[row, col].set_title(name)
        axes[row, col].legend(loc="best")
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()

#activation_tests()
#plot_activation_functions()