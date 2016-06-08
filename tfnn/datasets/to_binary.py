import numpy as np


def to_binary(data, inplace=False):
    unique, inverse = np.unique(data.ys, return_inverse=True)
    binary_classes = np.zeros((data.n_samples, len(unique)))
    for i in range(data.n_samples):
        binary_classes[i, inverse[i]] = 1

    if inplace:
        data.ys = binary_classes
    else:
        binary_data = data.copy()
        binary_data.ys = binary_classes
        return binary_data
