from utils import load


def training(dataset: str):
    df = load(dataset, header=None)
    print(df)
    print("Training entrypoint")
