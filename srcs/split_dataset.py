from srcs.utils import load, GREEN, CYAN, YELLOW, END
import pandas as pd
import os
import numpy as np


def create_data_directories(path: str, directories: str) -> None:
    """Create the necessary directory structure for data organization"""
    for directory in directories:
        os.makedirs(f"{path}{directory}", exist_ok=True)

def split_features(df: pd.DataFrame, train_size: float, val_size: float)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features into train, validation and test sets based on provided proportions."""
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    X_train = df.iloc[:train_end, :]
    X_val = df.iloc[train_end:val_end, :]
    X_test = df.iloc[val_end:, :]
    
    return X_train, X_val, X_test

def split_labels(y: pd.DataFrame, train_size: float, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labels into train, validation and test sets based on provided proportions."""
    n = len(y)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]
    
    return y_train, y_val, y_test

def create_sets(sets: tuple[pd.DataFrame], filepath: str, set_types: list[str]) -> None:
    """Convert dataframes to csv and save them in directories"""
    n_sets = len(sets) // 2
    for i in range(n_sets):
        set_type = set_types[i]

        X_file = f"data/processed/{set_type}/X_{set_type}.csv"
        print(f"{CYAN}{X_file}{GREEN} of shape {YELLOW}{sets[i].shape} {GREEN}successfully created{END}")
        y_file = f"data/processed/{set_type}/y_{set_type}.csv"
        print(f"{CYAN}{X_file}{GREEN} of shape {YELLOW}{sets[i + n_sets].shape} {GREEN}successfully created{END}")
        sets[i].to_csv(X_file, index=False, header=False)
        sets[i + n_sets].to_csv(y_file, index=False, header=False)  

def split_dataset(dataset_name: str, train_size: float = 0.7, val_size: float = 0.15) -> None:
    """Split dataset into train, validation and test sets"""
    if not 0 < train_size + val_size < 1:
        raise ValueError("Sum of train_size and val_size must be between 0 and 1")
    
    create_data_directories("data/processed/", ["train", "val", "test"])
    df = load(dataset_name, header=None)
    df = df.drop(columns=[0])  # drop ID column
    df.columns = range(df.shape[1])  # Rearrange column idx
    
    # replace and drop lines with invalid values (only 13 lines concerned)
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df[0] = df[0].map({'B': 0, 'M': 1})  # Benin = 0, Malin = 1

    y = df.iloc[:, 0].to_frame() # get labels in column 0
    
    df = df.drop(columns=[0]) # Drop labels X
    df.columns = range(df.shape[1])  # Rearrange columns idx
    
    X_train, X_val, X_test = split_features(df, train_size, val_size)
    y_train, y_val, y_test = split_labels(y, train_size, val_size)

    filepath = "data/processed/"
    set_types = ['train', 'val', 'test']
    create_data_directories(filepath, set_types)
    create_sets((X_train, X_val, X_test, y_train, y_val, y_test), filepath, set_types)
