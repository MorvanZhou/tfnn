import numpy as np


def sampled_batch(data, batch_size):
    sample_indexes = np.random.choice(data.n_samples, batch_size, replace=False)
    batch_xs = data.xs[sample_indexes, :]
    batch_ys = data.ys[sample_indexes, :]
    return [batch_xs, batch_ys]
