import numpy as np
import matplotlib.pyplot as plt

class WeightInitialiser:
    @staticmethod
    def he_normal(input_size, output_size):
        weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        return weight

    @staticmethod
    def he_uniform(input_size, output_size):
        limit = np.sqrt(6 / input_size)
        weight = np.random.uniform(-limit, limit, (input_size, output_size))
        return weight

    @staticmethod
    def glorot_uniform(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        weight = np.random.uniform(-limit, limit, (input_size, output_size))
        return weight

    @staticmethod
    def glorot_normal(input_size, output_size):
        stddev = np.sqrt(2 / (input_size + output_size))
        weight = np.random.randn(input_size, output_size) * stddev
        return weight

def plot_weight_initializations(input_size, output_size, seed=42):
    np.random.seed(seed)
    initializers = {
        "He Normal": WeightInitialiser.he_normal(input_size, output_size),
        "He Uniform": WeightInitialiser.he_uniform(input_size, output_size),
        "Glorot Normal": WeightInitialiser.glorot_normal(input_size, output_size),
        "Glorot Uniform": WeightInitialiser.glorot_uniform(input_size, output_size)
    }
    
    colors_activation = ["blue", "red", "orange", "lime"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, (name, weights) in enumerate(initializers.items()):
        row, col = divmod(i, 2)

        axes[row, col].hist(weights.flatten(), bins=30, alpha=0.6, color=colors_activation[i])
        axes[row, col].set_title(name)
        axes[row, col].grid(True)
    plt.tight_layout()
    plt.show()

#plot_weight_initializations(24, 24)