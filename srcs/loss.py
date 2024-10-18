import numpy as np


def log_loss(y_train, A_last):
    epsilon = 1e-8
    m = y_train.shape[0]
    loss = - 1 / m * np.sum(y_train * np.log(A_last + epsilon) + (1 - y_train) * np.log(1 - A_last + epsilon))
    return loss


# ---- TESTS ---- #

def log_loss_tests():
    print("--- Total uncertainty ---")
    y_train = np.array([1, 0])
    A_last = np.array([0.5, 0.5]) # 0.693.. = log(2)
    print(f"Result: {log_loss(y_train, A_last)}")
    print(f"Expected: ~ 0.693 = log(2)\
           because log(0.5) = -0.693..., -(0.693..) = log(2)")

    print("--- 100%% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([1, 0])
    print(f"Result: {log_loss(y_train, A_last)}")
    print(f"Expected: ~ 0 = as the loss tends to 0, -9.99..e-09 = -log(1 + 1e-8)")

    print("--- 99.999..%% accuracy")
    y_train = np.array([1, 0])
    A_last = np.array([1 - 1e-10, 1e-10])
    print(log_loss(y_train, A_last))

    print("--- 0%% accuracy ---")
    y_train = np.array([1, 0])
    A_last = np.array([0, 1])
    print(f"Result: {log_loss(y_train, A_last)}")
    print(f"Expected: 18.42.. = -log(1e-8)")


#log_loss_tests()
