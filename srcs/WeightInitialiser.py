import numpy as np

class WeightInitialiser:
    @staticmethod
    def he_normal(input_size, output_size):
        weight = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        return weight

    @staticmethod
    def he_uniform(input_size, output_size):
        limit = np.sqrt(6 / input_size)
        weight = np.random.uniform(-limit, limit, (input_size, output_size))
        return weight

    @staticmethod
    def glorot_uniform(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        weight = np.random.uniform(-limit, limit, (input_size, output_size))
        return weight

    @staticmethod
    def glorot_normal(input_size, output_size):
        stddev = np.sqrt(2 / (input_size + output_size))
        weight = np.random.randn(input_size, output_size) * stddev
        return weight
