import pandas as pd

YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
END = "\033[0m"

def load(path: str, header: str = "header") -> pd.DataFrame:
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path) if header is not None else pd.read_csv(path, header=None)
        print(f"Loading dataset of dimensions {df.shape}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        return exit(1)