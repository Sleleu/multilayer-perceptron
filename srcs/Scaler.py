import numpy as np

class Scaler:
    def __init__(self, method="z_score"):
        self.method = method
        self.mean = None
        self.scale = None
        self.min = None
        self.max = None
    
    def fit(self, X):
            self.mean = np.mean(X, axis=0)
            self.scale = np.std(X, axis=0)
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)

    def transform(self, X):
        if self.method == "z_score":
            if self.mean is None or self.scale is None:
                raise RuntimeError("You must fit the scaler before transforming data")
            return (X - self.mean) / (self.scale + 1e-15)
        elif self.method == 'minmax':
            if self.min is None or self.max is None:
                raise RuntimeError("You must fit the scaler before transforming data.")
            return (X - self.min) / ((self.max - self.min) + 1e-15)
        else:
            raise ValueError("Method must be either 'z_score' or 'minmax'.")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ------------------- TEST ------------------- #

if __name__ == "__main__":
    # Exemples de la doc StandardScaler: 
    # https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html
    print("\n   standard scaling test\n")
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    standard_scaler = Scaler(method="z_score")
    standard_scaler.fit(data)
    print(standard_scaler.mean)
    print(standard_scaler.transform(data))
    print(standard_scaler.transform([2, 2]))


    # Exemples de la doc MinMax scaler:
    # https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    print("\n   minmax scaling test\n")
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    minmax_scaler = Scaler(method="minmax")
    print(minmax_scaler.fit(data))
    print(minmax_scaler.max)
    print(minmax_scaler.transform(data))
    print(minmax_scaler.transform([2, 2]))
