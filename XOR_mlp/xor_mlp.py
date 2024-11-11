import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from srcs.Mlp import MLP
from srcs.EarlyStopping import EarlyStopping
from srcs.split_dataset import split_features, split_labels, create_data_directories, create_sets
from srcs.utils import load, YELLOW, GREEN, CYAN, END, get_accuracy
from srcs.training import save_model
from srcs.display import plot_learning_curves
import argparse
import pandas as pd
import numpy as np

def split_xor(dataset: str):
    # Source of dataset : https://www.kaggle.com/datasets/bipinmaharjan/xor-dataset
    df = load(dataset, header=None)
    X = df.iloc[:, :2]
    y = df.iloc[:, -1].to_frame()
    
    train_size = 0.7
    val_size = 0.15
    X_train, X_val, X_test = split_features(X, train_size, val_size)
    y_train, y_val, y_test = split_labels(y, train_size, val_size)
    
    filepath = "data/processed/"
    set_types = ['train', 'val', 'test']
    create_data_directories(filepath, set_types)
    create_sets((X_train, X_val, X_test, y_train, y_val, y_test), filepath, set_types)

def train_xor():
    X_train = pd.read_csv('data/processed/train/X_train.csv', header=None)
    y_train = pd.read_csv('data/processed/train/y_train.csv', header=None).to_numpy()
    X_val = pd.read_csv("data/processed/val/X_val.csv", header=None)
    y_val = pd.read_csv("data/processed/val/y_val.csv", header=None).to_numpy()

    print(f"{GREEN}x_train shape: {YELLOW}{X_train.shape}")
    print(f"{GREEN}x_val shape: {YELLOW}{X_val.shape}")
    print(f"\n\t{GREEN}TRAINING PHASE:{END}")

    early_stopping = EarlyStopping(patience=5)
    model = MLP(hidden_layer_sizes=[2],
                activation="sigmoid",
                output_activation="sigmoid",
                epochs=50,
                loss="binaryCrossentropy",
                output_layer_size=1,
                batch_size=64,
                learning_rate=0.01,
                random_seed=42,
                weight_initializer="GlorotUniform",
                solver="momentum"
                )
    best_W, best_b = model.fit(X_train, y_train, X_val, y_val, early_stopping)
    save_model(model, best_W, best_b)
    print(model)
    plot_learning_curves(model.train_losses, model.val_losses, model.train_accuracies, model.val_accuracies)

def predict():
    print(f"{GREEN}\t PREDICT PHASE:{END}\n")
    m_data = np.load("saved_model.npy", allow_pickle=True).item()
    model = MLP(
        hidden_layer_sizes=m_data['hidden_layer_sizes'],
        output_layer_size=m_data['output_layer_size'],
        activation=m_data['activation'],
        output_activation=m_data['output_activation'],
        loss=m_data['loss']
    )
    try:
        X = pd.read_csv("data/processed/test/X_test.csv", header=None)
    except Exception as e:
        print(f"{YELLOW}Error loading data: {e}{END}")
        exit(1)
        
    probabilities, _ = model.feed_forward(X, m_data['W'], m_data['b'])
    predictions = np.array(probabilities >= 0.5).astype(int)
    y_true = load("data/processed/test/y_test.csv", header=None).to_numpy()
    accuracy = get_accuracy(predictions, y_true)
    star = "🌟" if accuracy == 1.0 else "💩"
    print(f"{GREEN}Accuracy : {CYAN}{accuracy*100:.2f}% {star}{END}")

    results = pd.DataFrame({'Prediction': predictions.reshape(-1)})
    output_path = 'predictionsXOR.csv'
    results.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        required=True,
                        help="Path to the CSV file containing the data.")
    args = parser.parse_args()
    try:
        split_xor(args.dataset)
        train_xor()
        predict()
    except Exception as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        exit(1)
