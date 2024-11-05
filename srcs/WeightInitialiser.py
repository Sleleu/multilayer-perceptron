import numpy as np

class WeightInitialiser:
    @staticmethod
    def he_uniform(input_size, output_size):
        weight = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        biase = np.zeros((1, output_size))
        
        return weight, biase
