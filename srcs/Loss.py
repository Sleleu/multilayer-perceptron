import numpy as np
from srcs.utils import GREEN, END

class Loss:

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_true.shape[0]
        loss = - 1 / m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    @staticmethod
    def sparse_categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m

# ---- TESTS ---- #

def binary_cross_entropy_tests():
    print(f"\n{GREEN}=== Binary Cross Entropy Tests ==={END}")
    print("--- Total uncertainty ---")
    y_train = np.array([1, 0])
    A_last = np.array([0.5, 0.5]) # 0.693.. = log(2)
    print(f"Result: {Loss.binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: ~ 0.693 = log(2) because log(0.5) = -0.693..., -(0.693..) = log(2)")

    print("\n--- 100% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([1, 0])
    print(f"Result: {Loss.binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: ~ 0 = as the loss tends to 0")

    print("\n--- 99.999..% accuracy")
    y_train = np.array([1, 0])
    A_last = np.array([1 - 1e-10, 1e-10])
    print(f"Result: {Loss.binary_cross_entropy(y_train, A_last)}")
    print("Expected: Very close to 0")

    print("\n--- 0% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([0, 1])
    print(f"Result: {Loss.binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: Very high value")

def sparse_categorical_cross_entropy_tests():
    print(f"\n{GREEN}=== Sparse Categorical Cross Entropy Tests ==={END}")
    print("--- 100% accuracy ---")
    y_true = np.array([0, 1, 2])
    y_pred = np.array([
        [1.0, 0.0, 0.0],  # Predicts class 0 - 100% confidence
        [0.0, 1.0, 0.0],  # Predicts class 1 - 100% confidence
        [0.0, 0.0, 1.0],   # Predicts class 2 - 100% confidence
    ])
    print(f"Result: {Loss.sparse_categorical_cross_entropy(y_true, y_pred)}")
    print("Expected: ~ 0 (perfect predictions)")

    print("\n--- 50% accuracy ---")
    y_true = np.array([0, 1, 2])
    y_pred = np.array([
        [0.5, 0.3, 0.2],  # Predicts class 0 - 50% confidence
        [0.3, 0.5, 0.2],  # Predicts class 1 - 50% confidence
        [0.2, 0.3, 0.5],   # Predicts class 2 - 50% confidence
    ])
    print(f"Result: {Loss.sparse_categorical_cross_entropy(y_true, y_pred)}")
    print("Expected: ~ 0.693 (log(2))")

def categorical_cross_entropy_tests():
    print(f"\n{GREEN}=== Categorical Cross Entropy Tests ==={END}")
    print("--- 100% accuracy ---")
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    y_pred = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    print(f"Result: {Loss.categorical_cross_entropy(y_true, y_pred)}")
    print("Expected: ~ 0 (perfect predictions)")

    print("\n--- 50% accuracy ---")
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    y_pred = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])
    print(f"Result: {Loss.categorical_cross_entropy(y_true, y_pred)}")
    print("Expected: ~ 0.693 (log(2))")

if __name__ == "__main__":
    binary_cross_entropy_tests()
    sparse_categorical_cross_entropy_tests()
    categorical_cross_entropy_tests()
