import numpy as np


def binary_cross_entropy(y_train, y_pred):
    epsilon = 1e-15
    m = y_train.shape[0]
    loss = - 1 / m * np.sum(y_train * np.log(y_pred + epsilon) + (1 - y_train) * np.log(1 - y_pred + epsilon))
    return loss

def sparse_categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m

# ---- TESTS ---- #

def binary_cross_entropy_tests():
    print("--- Total uncertainty ---")
    y_train = np.array([1, 0])
    A_last = np.array([0.5, 0.5]) # 0.693.. = log(2)
    print(f"Result: {binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: ~ 0.693 = log(2)\
           because log(0.5) = -0.693..., -(0.693..) = log(2)")

    print("--- 100%% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([1, 0])
    print(f"Result: {binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: ~ 0 = as the loss tends to 0, -9.99..e-09 = -log(1 + 1e-8)")

    print("--- 99.999..%% accuracy")
    y_train = np.array([1, 0])
    A_last = np.array([1 - 1e-10, 1e-10])
    print(binary_cross_entropy(y_train, A_last))

    print("--- 0%% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([0, 1])
    print(f"Result: {binary_cross_entropy(y_train, A_last)}")
    print(f"Expected: 18.42.. = -log(1e-8)")


#log_loss_tests()
