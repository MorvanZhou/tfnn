import numpy as np


def shuffle(data, inplace=False):
    n_x = data.xs.shape[1]
    xs_ys = np.hstack((data.xs, data.ys))
    np.random.shuffle(xs_ys)
    xs = xs_ys[:, :n_x]
    ys = xs_ys[:, n_x:]
    if inplace:
        data.xs, data.ys = xs, ys
    else:
        shuffled_data = data.copy()
        shuffled_data.xs, shuffled_data.ys = xs, ys
        return shuffled_data