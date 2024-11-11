import pandas as pd
import numpy as np

YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
MAGENTA = "\033[1;34m"
END = "\033[0m"

def get_accuracy(y_pred, y_true):
    if len(y_pred.shape) == 1:
        raise ValueError("get_accuracy(): parameters have invalid shape: (m,)")
    if y_pred.shape[1] == 1:
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true.reshape(-1, 1))
    else:
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)


def load(path: str, header: str = "header") -> pd.DataFrame:
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path) if header is not None \
                               else pd.read_csv(path, header=None)
        print(f"{GREEN}Loading dataset of dimensions {YELLOW}{df.shape}{END}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        return exit(1)
