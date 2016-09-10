import numpy as np


def shuffle(data):
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data.data)
    return shuffled_data