import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0) * 1

def softmax(Z):
    assert len(Z.shape) == 2

    Z_max = np.max(Z, axis=1, keepdims=1)
    e_x = np.exp(Z - Z_max)
    div = np.sum(e_x, axis=1, keepdims=1)
    return e_x / div

def softmax_tests():
    print("=== SOFTMAX TESTS ===")
    Z = np.array([[1, 3, 2.5, 5, 4, 2]])
    A = softmax(Z)
    print("--- Test from wikipedia ---")
    print(f"A: {A}")
    print(f"Sum of A: {np.sum(A)}")

    Z = np.array([[1, 2, 3, 6],  # sample 1
               [2, 4, 5, 6],  # sample 2
               [1, 2, 3, 6]]) # sample 1 again(!)
    A = softmax(Z)
    print("\n--- Test from stackoverflow, broadcasting ---")
    print(f"A: {A}")
    print(f"Sum of A: {np.sum(A, axis=1)}")

# link : https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
#softmax_tests()