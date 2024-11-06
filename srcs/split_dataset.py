from srcs.utils import load, YELLOW, END
import pandas as pd

def split_features(df: pd.DataFrame, split_value: int)-> tuple[pd.DataFrame]:
    # Drop les labels du df d'origine et créer un dataset X_train X_test
    df = df.drop(columns=[0]) # Retirer les IDs
    df.columns = range(df.shape[1]) # Réarranger indexs des colonnes
    X_train = df.iloc[: split_value, :]
    X_test = df.iloc[split_value :, :]
    return (X_train, X_test)

def split_labels(y: pd.DataFrame, split_value: int)-> tuple[pd.DataFrame]:
    """Take a DataFrame of labels, and split into 2 parts, separated at 'split_value'"""
    y_train = y.iloc[: split_value]
    y_test = y.iloc[split_value :]
    return (y_train, y_test)

def create_sets(sets: tuple[pd.DataFrame], names: list[str])-> None:
    """Convert a tuple of dataframes in csv files"""
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, OSError)
    try: 
        for i, set in enumerate(sets):
            set.to_csv(f"data/{names[i]}", index=False)
    except HANDLED_ERRORS as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        return exit(1)


def split_dataset(dataset_name: str)-> None:
    df = load(dataset_name, header=None)
    df = df.drop(columns=[0]) # Retirer les IDs
    df.columns = range(df.shape[1]) # Réarranger indexs des colonnes
    df[0] = df[0].map({'B': 0, 'M': 1}) # Benin = 0, Malin = 1

    # Récupérer les 0,1 colonne 0 et les mettre dans un dataframe y de dimension (n[0], 1)
    y = df.iloc[:, 0].to_frame()
    split_value = 400  #df.shape[0] // 2 # 284, permet de split en deux

    X_train, X_test = split_features(df, split_value)
    y_train, y_test = split_labels(y, split_value)
    names = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    create_sets((X_train, X_test, y_train, y_test), names)
