from utils import load


def split_dataset(dataset_name: str):
    df = load(dataset_name, header=None)
    df = df.drop(columns=[0])
    df.columns = range(df.shape[1])
    df[0] = df[0].map({'B': 0, 'M': 1})
    print(df)
