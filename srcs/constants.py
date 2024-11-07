from srcs.Activation import Activation
from srcs.Loss import Loss

ACTIVATIONS_FUNCTIONS: dict[str, tuple] = {
    "sigmoid": (Activation.sigmoid, Activation.sigmoid_derivative),
    "relu": (Activation.relu, Activation.relu_derivative),
    "leakyrelu": (Activation.leaky_relu, Activation.leaky_relu_derivative),
    "tanh": (Activation.tanh, Activation.tanh_derivative)
}

OUTPUT_ACTIVATIONS: dict = {
    "softmax": Activation.softmax,
    "sigmoid": Activation.sigmoid
}

LOSS_FUNCTIONS: dict = {
    "sparseCategoricalCrossentropy": Loss.sparse_categorical_cross_entropy,
    "binaryCrossentropy": Loss.binary_cross_entropy,
    "categoricalCrossentropy": Loss.categorical_cross_entropy
}

